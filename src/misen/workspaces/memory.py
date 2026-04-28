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

import logging
import shutil
import tempfile
import threading
import weakref
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, TextIO

from misen.exceptions import LockUnavailableError
from misen.utils.hashing import ResultHash
from misen.workspace import Workspace

if TYPE_CHECKING:
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
        if key in self._paths:
            return
        target = self._payload_path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
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
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


class InMemoryWorkspace(Workspace):
    """Workspace backed by process-local memory and a temp directory.

    Hash indices and locks live in Python objects that vanish with the
    workspace. Result payloads, work directories, and logs are written
    under ``directory``; when ``directory`` is left as ``None`` the
    workspace allocates a fresh temp directory and removes it on
    :meth:`close` or when the workspace is finalized.
    """

    directory: str | None = None

    def __post_init__(self) -> None:
        """Create directory layout and initialize in-memory caches."""
        if self.directory is None:
            self._directory = Path(tempfile.mkdtemp(prefix="misen-mem-"))
            self._owns_directory = True
        else:
            self._directory = Path(self.directory)
            self._directory.mkdir(parents=True, exist_ok=True)
            self._owns_directory = False

        self._locks: dict[tuple[str, str], _ThreadLock] = {}
        self._locks_table_lock = threading.Lock()

        self.get_temp_dir().mkdir(parents=True, exist_ok=True)
        (self._directory / "work").mkdir(parents=True, exist_ok=True)
        (self._directory / "task_logs").mkdir(parents=True, exist_ok=True)
        (self._directory / "job_logs").mkdir(parents=True, exist_ok=True)

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
            self._finalizer.detach()
            _cleanup_directory(self._directory)

    def lock(self, namespace: Literal["task", "result"], key: str) -> LockLike:
        """Return per-(namespace, key) in-process lock."""
        with self._locks_table_lock:
            return self._locks.setdefault((namespace, key), _ThreadLock())

    def get_temp_dir(self) -> Path:
        """Return workspace temporary directory path."""
        return self._directory / "tmp"

    def _get_work_dir(self, task: Task) -> Path:
        """Return stable working directory for a task."""
        key_str = task.resolved_hash(workspace=self).b32()
        d = self._directory / "work" / key_str
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _task_log_dir(self, task: Task) -> tuple[Path, str]:
        # Logs are keyed by resolved_hash; resolution requires every dependency's
        # result hash to be cached, otherwise this raises ``CacheError``.
        key_str = task.resolved_hash(workspace=self).b32()
        log_dir = self._directory / "task_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir, key_str

    def get_task_log(self, task: Task, job_id: str | None = None) -> Path:
        """Return path where ``task``'s log for ``job_id`` should be written."""
        log_dir, key_str = self._task_log_dir(task)
        return log_dir / f"{key_str}_{job_id or '0'}.log"

    def read_task_log(self, task: Task, job_id: str | None = None) -> TextIO:
        """Open a previously-written task log for reading."""
        log_dir, key_str = self._task_log_dir(task)
        if job_id is None:
            matches = sorted(log_dir.glob(f"{key_str}_*.log"), key=lambda p: p.stat().st_mtime)
            if not matches:
                msg = f"No logs found for {key_str} in {log_dir}"
                raise FileNotFoundError(msg)
            log_path = matches[-1]
        else:
            log_path = log_dir / f"{key_str}_{job_id}.log"
        return log_path.open("r", buffering=1)
