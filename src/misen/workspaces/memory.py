"""In-memory workspace implementation.

This backend keeps:

- task/resolved/result hash indices in process-local Python dicts
- result payload directories on a per-workspace temp directory
- locks as in-process :mod:`threading` primitives
- task/job logs under the same temp directory

Suitable for :class:`misen.executors.in_process.InProcessExecutor`, where
every task runs in the same Python process. Not suitable for executors
that spawn worker processes (e.g.
:class:`misen.executors.local.LocalExecutor`,
:class:`misen.executors.slurm.SlurmExecutor`): the hash caches and locks
live in process-local memory, so workers in other interpreters cannot
observe cached results or coordinate via the runtime lock.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import tempfile
import threading
import weakref
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from misen.exceptions import LockUnavailableError, StorageError
from misen.utils.fsync import atomic_write_bytes as _atomic_write_bytes
from misen.utils.hashing import ResultHash
from misen.workspace import Workspace, _storage_errors

if TYPE_CHECKING:
    from collections.abc import Callable

    from misen.tasks import Task
    from misen.utils.locks import LockLike

logger = logging.getLogger(__name__)


class _ThreadLock:
    """Single-process lock implementing :class:`misen.utils.locks.LockLike`."""

    __slots__ = ("_lock",)

    _lock: threading.Lock

    def __init__(self) -> None:
        # Non-reentrant matches the runtime contract ("at most one active
        # execution per resolved key") and the file-lock semantics of NFSLock.
        self._lock = threading.Lock()

    def acquire(self, *, blocking: bool = True, timeout: int | None = None) -> None:
        """Acquire lock, optionally waiting up to ``timeout`` seconds."""
        # threading.Lock.acquire uses timeout=-1 for "wait forever"; the
        # workspace API uses None for the same meaning.
        wait = -1 if timeout is None else timeout
        if not self._lock.acquire(blocking=blocking, timeout=wait):
            msg = "Could not acquire in-memory lock."
            raise LockUnavailableError(msg)

    def release(self) -> None:
        """Release the underlying lock."""
        self._lock.release()

    @contextmanager
    def context(self, *, blocking: bool = True, timeout: int | None = None) -> Iterator[Self]:
        """Context manager that acquires/releases the lock."""
        self.acquire(blocking=blocking, timeout=timeout)
        try:
            yield self
        finally:
            self.release()

    def is_locked(self) -> bool:
        """Return whether the lock is currently held."""
        return self._lock.locked()


class _MemoryResultStore(MutableMapping[ResultHash, Path]):
    """In-memory mapping of result hashes to payload directories.

    :meth:`__setitem__` adopts the directory passed by
    :class:`misen.workspace.ResultMap` by moving it under ``root_dir``,
    so the caller's temp-dir cleanup does not delete the payload.
    """

    __slots__ = ("_paths", "_root")

    _root: Path
    _paths: dict[ResultHash, Path]

    def __init__(self, root_dir: Path) -> None:
        self._root = root_dir
        self._paths = {}

    def _payload_path(self, key: ResultHash) -> Path:
        return self._root / key.b32()

    def __contains__(self, key: object) -> bool:
        return isinstance(key, ResultHash) and key in self._paths

    def __getitem__(self, key: ResultHash) -> Path:
        return self._paths[key]

    def __setitem__(self, key: ResultHash, value: Path) -> None:
        self.commit(key, value, before_commit=lambda: None)

    def commit(self, key: ResultHash, value: Path, *, before_commit: Callable[[], None]) -> None:
        """Move a result into place after a caller-supplied ownership check."""
        if key in self._paths:
            return
        target = self._payload_path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        before_commit()
        shutil.move(value, target)
        self._paths[key] = target

    def __delitem__(self, key: ResultHash) -> None:
        path = self._paths.pop(key)
        if path.exists():
            shutil.rmtree(path)

    def __iter__(self) -> Iterator[ResultHash]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)


def _cleanup_directory(path: Path) -> None:
    """Remove ``path`` recursively; quiet on missing or partially-removed trees."""
    with contextlib.suppress(OSError):
        shutil.rmtree(path)


class InMemoryWorkspace(Workspace):
    """Workspace backed by process-local memory and a temp directory.

    Hash indices and locks live in Python objects that vanish with the
    workspace. Result payloads, scratch directories, and logs are written
    under ``directory``; when ``directory`` is left as ``None`` the
    workspace allocates a fresh temp directory and removes it on
    :meth:`close` or when the workspace is finalized.
    """

    directory: str | None = None

    def __post_init__(self) -> None:
        """Create directory layout and initialize in-memory caches."""
        self._owns_directory = False
        try:
            if self.directory is None:
                self._directory = Path(tempfile.mkdtemp(prefix="misen-mem-"))
                self._owns_directory = True
            else:
                self._directory = Path(self.directory)
                self._directory.mkdir(parents=True, exist_ok=True)
                self._owns_directory = False

            self.get_temp_dir().mkdir(parents=True, exist_ok=True)
            (self._directory / "scratch").mkdir(parents=True, exist_ok=True)
            (self._directory / "task_logs").mkdir(parents=True, exist_ok=True)
            (self._directory / "job_logs").mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            if self._owns_directory:
                _cleanup_directory(self._directory)
            location = self.directory or "an automatic temporary directory"
            msg = f"Could not initialize in-memory workspace at {location}: {exc}"
            raise StorageError(msg) from exc

        self._locks: dict[tuple[str, str], _ThreadLock] = {}
        self._locks_table_lock = threading.Lock()

        super()._post_init(
            resolved_hash_cache={},
            result_hash_cache={},
            result_store=_MemoryResultStore(self._directory / "results"),
        )

        if self._owns_directory:
            # Reclaim the auto-created tempdir when the workspace is GC'd.
            self._finalizer = weakref.finalize(self, _cleanup_directory, self._directory)

        logger.info("Initialized InMemoryWorkspace at %s.", self._directory)

    def close(self) -> None:
        """Remove the auto-created temp directory. Idempotent.

        No-op when the workspace was constructed with an explicit
        ``directory`` argument; the caller owns that directory.
        """
        if self._owns_directory:
            try:
                shutil.rmtree(self._directory)
            except FileNotFoundError:
                pass
            except OSError as exc:
                msg = f"Could not close in-memory workspace at {self._directory}: {exc}"
                raise StorageError(msg) from exc
            self._finalizer.detach()
            self._owns_directory = False

    def lock(self, namespace: Literal["task", "result"], key: str) -> LockLike:
        """Return per-(namespace, key) in-process lock."""
        with self._locks_table_lock:
            return self._locks.setdefault((namespace, key), _ThreadLock())

    def get_temp_dir(self) -> Path:
        """Return workspace temporary directory path."""
        return self._directory / "tmp"

    def publish_snapshot(self, key: str, staged_dir: Path) -> None:
        """Publish a snapshot into the workspace's temporary directory."""
        final = self._directory / "snapshots" / key
        with _storage_errors(f"Could not publish snapshot {key!r} to {self._directory}"):
            if not final.exists():
                final.parent.mkdir(parents=True, exist_ok=True)
                try:
                    staged_dir.rename(final)
                except OSError:
                    if not final.is_dir():
                        raise

    def fetch_snapshot(self, key: str) -> Path:
        """Return a locally published snapshot directory."""
        path = self._directory / "snapshots" / key
        with _storage_errors(f"Could not inspect snapshot {key!r} in {path.parent}"):
            available = path.is_dir()
        if not available:
            msg = f"No snapshot published under key {key!r} in {path.parent}."
            raise FileNotFoundError(msg)
        return path

    def put_job_file(self, submission_id: str, name: str, data: bytes) -> str:
        """Store an owner-only submission file and return its path."""
        self._validate_job_file_name(name)
        path = self._directory / "job_files" / submission_id / name
        with _storage_errors(f"Could not persist job file {name!r} for submission {submission_id!r}"):
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(path, data)
            path.chmod(0o600)
        return str(path)

    def read_job_file(self, submission_id: str, name: str) -> bytes:
        """Read one submission file without hiding a not-yet-published file."""
        self._validate_job_file_name(name)
        path = self._directory / "job_files" / submission_id / name
        with _storage_errors(f"Could not read job file {name!r}", passthrough=(FileNotFoundError,)):
            return path.read_bytes()

    def bootstrap_transport(self) -> None:
        """Use directly visible paths for this process-local workspace."""

    def _get_scratch_dir(self, task: Task) -> Path:
        """Return stable scratch directory for a task."""
        key_str = task.resolved_hash(workspace=self).b32()
        d = self._directory / "scratch" / key_str
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _task_log_dir(self, task: Task) -> tuple[Path, str]:
        """Flat task-log directory behind the base path-backed log methods."""
        key_str = task.resolved_hash(workspace=self).b32()
        log_dir = self._directory / "task_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir, key_str
