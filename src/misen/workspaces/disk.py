"""Disk-backed workspace implementation.

This backend persists:

- task/resolved/result hash indices as one atomically-written file per key
- result payload directories on disk
- lock files for cross-process coordination (NFS-compatible)
- task/job logs for runtime observability

The design prioritizes deterministic paths, write-once result materialization,
and lock-based safety for concurrent producers.
"""

from __future__ import annotations

import binascii
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Literal, Self, TypeVar

from misen.exceptions import LockUnavailableError, StorageError
from misen.utils.fsync import atomic_write_bytes as _atomic_write_bytes
from misen.utils.fsync import fsync_dir as _fsync_dir
from misen.utils.fsync import fsync_file as _fsync_file
from misen.utils.hashing import Hash, ResolvedTaskHash, ResultHash, TaskHash
from misen.utils.locks import LockLike, NFSLock
from misen.workspace import _hash_mapping_type, _PathWorkspace, _storage_errors

if TYPE_CHECKING:
    from collections.abc import Callable

    from misen.tasks import Task

KT = TypeVar("KT", bound=Hash)
VT = TypeVar("VT", bound=Hash)
logger = logging.getLogger(__name__)


def _fsync_tree(root: Path) -> None:
    """Fsync every file and directory under ``root`` so the whole tree is durable.

    Serializer backends (parquet, numpy, torch, ...) and ``manifest.json`` only
    flush their bytes to the kernel on close. On NFS that reaches the server
    (so close-to-open readers see it) but is not COMMITted to stable storage,
    and a freshly created file's directory entry is likewise not yet durable.
    Without this, a crash could publish a result directory whose files are
    empty/partial or whose entries are missing -- the dangling-payload state
    :func:`misen.utils.task_utils.save_task_result` is built to avoid.

    The walk is bottom-up so each file's contents and each child directory's
    entries are made durable before the parent directory that names them. This
    costs one COMMIT per file and per directory, paid once at result-commit
    time (not in any hot loop), in exchange for a crash-consistent payload.

    Args:
        root: Directory whose entire contents should be made durable.
    """
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        base = Path(dirpath)
        for name in filenames:
            file_path = base / name
            if file_path.is_symlink() or not file_path.is_file():
                continue
            _fsync_file(file_path)
        _fsync_dir(base)


# Both stores shard keys by the first two chars of ``Hash.b32`` (see hash_types).
# The ``[A-Z2-7]`` charset is exactly the unpadded base32 alphabet, so the glob
# matches only canonical key names and skips the leading-dot temp/trash entries
# an in-flight write or delete may leave behind.
_B32_LEN = len(ResultHash(0).b32())
_B32_SHARD_GLOB = f"{'[A-Z2-7]' * 2}/{'[A-Z2-7]' * _B32_LEN}"


class FileKVMapping(MutableMapping[KT, VT], Generic[KT, VT]):
    """Typed hash->hash mapping stored as one file per key, NFS-tolerant.

    Each entry is a single file written with an atomic ``os.replace`` and read
    with a fresh ``open`` (close-to-open semantics), so a reader on another
    host never observes a torn or stale value the way a held-open LMDB
    ``mmap`` can on NFS. Keys and values must be
    :class:`misen.utils.hashing.Hash` subclasses supporting
    ``b32``/``encode``/``decode``.

    The store is lock-free: every value is a deterministic function of its key
    (resolved/result hashes are pure functions of their inputs), so concurrent
    writers emit identical bytes and ``os.replace`` last-writer-wins is always
    safe. A freshly written key may briefly read as absent on another host
    (NFS negative-dentry caching); callers treat that as a cache miss and
    recompute -- at worst a redundant computation, never a wrong value.
    """

    _key_type: type[KT]
    _value_type: type[VT]
    __slots__ = ("_directory",)

    def __class_getitem__(cls, item: tuple[type[KT], type[VT]]) -> type[Self]:
        """Parameterize the mapping with concrete key/value hash types."""
        return _hash_mapping_type(cls, item)

    def __init__(self, directory: Path) -> None:
        """Initialize the file-backed mapping rooted at ``directory``.

        Args:
            directory: Root directory holding the sharded per-key files.
        """
        if not hasattr(self, "_key_type") or not hasattr(self, "_value_type"):
            msg = "Construct as FileKVMapping[KeyType, ValueType](...)"
            raise TypeError(msg)
        self._directory = directory

    def _key_path(self, key: KT) -> Path:
        """Return the canonical sharded file path for ``key``."""
        _key = key.b32()
        return self._directory / _key[:2] / _key

    def __getitem__(self, key: KT) -> VT:
        """Return the value stored for ``key``.

        Raises:
            KeyError: If the key is not present.
        """
        try:
            return self._value_type.decode(self._key_path(key).read_bytes())
        except FileNotFoundError as e:
            raise KeyError(key) from e
        except ValueError as exc:
            msg = f"Stored value for {key!r} is corrupt or incompatible."
            raise StorageError(msg) from exc
        except OSError as exc:
            msg = f"Could not read stored value for {key!r}: {exc}"
            raise StorageError(msg) from exc

    def __setitem__(self, key: KT, value: VT) -> None:
        """Atomically write the value for ``key``, overwriting any prior value.

        Uses the shared durable-replace sequence (temp file, fsync, atomic
        rename, directory fsync), so a concurrent reader sees either the
        old value or the new one, never a partial write, and the rename
        survives a crash.

        Args:
            key: Hash key.
            value: Hash value.
        """
        path = self._key_path(key)
        with _storage_errors(f"Could not persist stored value for {key!r}"):
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(path, value.encode())

    def commit(self, key: KT, value: VT, *, before_commit: Callable[[], None]) -> None:
        """Create a fenced, write-once index entry without stale overwrites."""
        path = self._key_path(key)
        encoded = value.encode()
        with _storage_errors(f"Could not persist stored value for {key!r}"):
            path.parent.mkdir(parents=True, exist_ok=True)
            if _atomic_write_bytes(path, encoded, before_commit=before_commit, overwrite=False):
                return
            if path.read_bytes() != encoded:
                msg = f"Another writer committed a different value for {key!r}."
                raise LockUnavailableError(msg)

    def __delitem__(self, key: KT) -> None:
        """Remove the value stored for ``key``.

        Raises:
            KeyError: If the key is not present.
        """
        path = self._key_path(key)
        try:
            path.unlink()
        except FileNotFoundError as e:
            raise KeyError(key) from e
        except OSError as exc:
            msg = f"Could not delete stored value for {key!r}: {exc}"
            raise StorageError(msg) from exc
        with _storage_errors(f"Could not durably delete stored value for {key!r}"):
            _fsync_dir(path.parent)

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` has a stored value.

        Args:
            key: Candidate key.

        Returns:
            ``True`` if ``key`` is of the expected type and present on disk.
        """
        if not isinstance(key, self._key_type):
            return False
        with _storage_errors(f"Could not inspect stored value for {key!r}"):
            return self._key_path(key).is_file()

    def __iter__(self) -> Iterator[KT]:
        """Iterate over the typed keys currently stored.

        The ``[A-Z2-7]`` glob and shard-prefix check skip the leading-dot
        ``.tmp`` files an in-flight :meth:`__setitem__` may leave behind, plus
        any foreign file.
        """
        with _storage_errors(f"Could not list stored values in {self._directory}"):
            for p in self._directory.glob(_B32_SHARD_GLOB):
                if not p.is_file() or p.name[:2] != p.parent.name:
                    continue
                try:
                    yield self._key_type.from_b32(p.name)
                except (binascii.Error, ValueError):
                    continue

    def __len__(self) -> int:
        """Return the number of stored keys."""
        return sum(1 for _ in self)


class DiskResultStore(MutableMapping[ResultHash, Path]):
    """Mapping of result hashes to payload directories on disk."""

    __slots__ = ("directory",)

    directory: Path

    def __init__(self, directory: Path) -> None:
        """Initialize result store rooted at ``directory``.

        Args:
            directory: Root directory for sharded result payloads.
        """
        self.directory = directory

    def _result_dir_path(self, key: ResultHash) -> Path:
        """Return canonical sharded directory for a result hash."""
        _key = key.b32()
        return self.directory / _key[:2] / _key

    def __contains__(self, key: object) -> bool:
        """Return whether result payload exists on disk."""
        if not isinstance(key, ResultHash):
            return False
        with _storage_errors(f"Could not inspect stored result {key.b32()}"):
            return self._result_dir_path(key).exists()

    def __getitem__(self, key: ResultHash) -> Path:
        """Return the directory for a result hash.

        Raises:
            KeyError: If the result directory is missing.
        """
        result_dir_path = self._result_dir_path(key)
        with _storage_errors(f"Could not inspect stored result {key.b32()}"):
            exists = result_dir_path.exists()
        if not exists:
            raise KeyError(key)
        return result_dir_path

    def __setitem__(self, key: ResultHash, value: Path) -> None:
        """Publish a serialized payload directory."""
        self.commit(key, value, before_commit=lambda: None)

    def commit(self, key: ResultHash, value: Path, *, before_commit: Callable[[], None]) -> None:
        """Atomically publish a serialized payload directory.

        The payload tree is fsync'd in full (:func:`_fsync_tree`) and then moved
        into its final, content-addressed location with a single ``os.rename`` --
        atomic within one filesystem -- so a concurrent reader observes either no
        directory or the fully populated one, never a partially-written payload.
        The contents fsync matters because serializer backends only flush their
        files to the kernel on close (which on NFS reaches the server but is not
        COMMITted to stable storage); fsyncing first closes the window where a
        crash could publish a directory full of empty or partial files. The
        parent directory is then fsync'd so the rename survives a crash, which
        lets :func:`misen.utils.task_utils.save_task_result` treat the payload as
        durably present before it writes the ``result_hash`` pointer.

        ``value`` is the workspace temp dir and the store both live under the
        workspace root (one filesystem), so the rename never silently falls back
        to a non-atomic copy; an ``OSError`` here surfaces a genuine
        misconfiguration rather than degrading the atomicity guarantee.

        Args:
            key: Result hash.
            value: Temporary directory containing the serialized payload.
            before_commit: Ownership check run beside the atomic rename.
        """
        result_dir_path = self._result_dir_path(key)
        with _storage_errors(f"Could not persist stored result {key.b32()}"):
            if not result_dir_path.exists():
                result_dir_path.parent.mkdir(parents=True, exist_ok=True)
                _fsync_tree(value)  # flush payload contents+entries before the publish rename
                before_commit()
                os.rename(value, result_dir_path)  # noqa: PTH104  -- explicit atomic rename, no copy fallback
                _fsync_dir(result_dir_path.parent)

    def __delitem__(self, key: ResultHash) -> None:
        """Delete a result directory.

        Raises:
            KeyError: If the directory is missing.
        """
        result_dir_path = self._result_dir_path(key)
        if not result_dir_path.exists():
            raise KeyError(key)
        try:
            # atomic deletion
            trash_dir = Path(
                tempfile.mkdtemp(dir=result_dir_path.parent, prefix=f"{result_dir_path.name}.", suffix=".trash")
            )
            shutil.move(result_dir_path, trash_dir)
            _fsync_dir(result_dir_path.parent)
            shutil.rmtree(trash_dir)
            _fsync_dir(trash_dir.parent)
        except KeyError:
            raise
        except OSError as exc:
            msg = f"Could not delete stored result {key.b32()}: {exc}"
            raise StorageError(msg) from exc

    def __iter__(self) -> Iterator[ResultHash]:
        """Iterate over stored result hashes.

        Payloads are sharded directories named by :meth:`ResultHash.b32`; the
        ``[A-Z2-7]`` glob and shard-prefix check skip the in-flight ``.trash``
        dir a concurrent :meth:`__delitem__` may leave behind, plus any foreign
        entry.
        """
        with _storage_errors(f"Could not list stored results in {self.directory}"):
            for p in self.directory.glob(_B32_SHARD_GLOB):
                if not p.is_dir() or p.name[:2] != p.parent.name:
                    continue
                try:
                    yield ResultHash.from_b32(p.name)
                except (binascii.Error, ValueError):
                    continue

    def __len__(self) -> int:
        """Return number of stored results."""
        return sum(1 for _ in self)


class DiskWorkspace(_PathWorkspace):
    """Workspace implementation backed by local/NFS-accessible directories."""

    directory: str = ".misen"

    def __post_init__(self) -> None:
        """Create directory layout and initialize persistent caches."""
        # Lock to the absolute path at construction time so the msgpack-encoded
        # field carries this CWD's resolution. Workers (esp. SLURM) decode in a
        # CWD that doesn't always match the orchestrator's, so a relative
        # directory would resolve to a different tree on the worker.
        self.directory = str(Path(self.directory).absolute())
        self._directory_path = Path(self.directory)
        with _storage_errors(f"Could not initialize disk workspace at {self._directory_path}"):
            self._directory_path.mkdir(exist_ok=True)
            self.get_temp_dir().mkdir(parents=True, exist_ok=True)
            (self._directory_path / "scratch").mkdir(parents=True, exist_ok=True)
            (self._directory_path / "task_logs").mkdir(parents=True, exist_ok=True)
            (self._directory_path / "job_logs").mkdir(parents=True, exist_ok=True)
            (self.get_temp_dir() / "task_locks").mkdir(parents=True, exist_ok=True)
            (self.get_temp_dir() / "result_locks").mkdir(parents=True, exist_ok=True)

        super()._post_init(
            resolved_hash_cache=FileKVMapping[TaskHash, ResolvedTaskHash](self._directory_path / "resolved_hash_cache"),
            result_hash_cache=FileKVMapping[ResolvedTaskHash, ResultHash](self._directory_path / "result_hash_cache"),
            result_store=DiskResultStore(self._directory_path / "results"),
        )
        logger.info("Initialized DiskWorkspace at %s.", self._directory_path)

    def close(self) -> None:
        """Release workspace resources. Idempotent no-op for the file backend.

        The file-backed hash caches hold no open handles or locks, so there is
        nothing to release. The method is retained (and kept idempotent) so
        callers can wrap any workspace in ``contextlib.closing(workspace)``
        uniformly. Note that workspace instances are memoized by constructor
        kwargs, so re-constructing the same workspace returns the same instance
        within the process.
        """

    def lock(self, namespace: Literal["task", "result"], key: str) -> LockLike:
        """Return NFS-backed lock for task/result namespaces.

        Args:
            namespace: Lock namespace.
            key: Lock key.

        Returns:
            Lock-like object.

        Notes:
            Task-namespace locks back the cacheable-task runtime exclusivity
            guarantee for a given workspace and resolved task key.
        """
        return NFSLock(
            lockfile=(self.get_temp_dir() / f"{namespace}_locks" / f"{key}.lock"),
            lifetime=30,
            refresh_interval=20,
        )

    def get_temp_dir(self) -> Path:
        """Return workspace temporary directory path."""
        return self._directory_path / "tmp"

    def publish_snapshot(self, key: str, staged_dir: Path) -> None:
        """Publish a staged snapshot tree by atomic rename + durable marker.

        Extends the base path-backed publication with NFS crash safety:
        the tree is fsync'd before the rename, and the fsync'd
        ``<key>.complete`` marker is the commit point (payload before
        pointer). A tree without a marker is a crashed publisher's residue,
        and — because entries are content-addressed and the rename was
        atomic — that residue is a *complete* tree, so a later publisher
        may simply re-commit the marker.
        """
        with _storage_errors(f"Could not publish snapshot {key!r} to {self._directory_path}"):
            snapshots_dir = self._directory_path / "snapshots"
            if (snapshots_dir / f"{key}.complete").is_file() and (snapshots_dir / key).is_dir():
                return
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            _fsync_tree(staged_dir)
            final = snapshots_dir / key
            try:
                staged_dir.rename(final)
            except OSError:
                if not final.is_dir():
                    raise
            _fsync_dir(snapshots_dir)
            _atomic_write_bytes(snapshots_dir / f"{key}.complete", b"complete\n")

    def fetch_snapshot(self, key: str) -> Path:
        """Return a marker-committed snapshot directory."""
        snapshots_dir = self._directory_path / "snapshots"
        path = snapshots_dir / key
        with _storage_errors(f"Could not inspect snapshot {key!r} in {snapshots_dir}"):
            available = (snapshots_dir / f"{key}.complete").is_file() and path.is_dir()
        if not available:
            msg = f"No snapshot published under key {key!r} in {snapshots_dir}."
            raise FileNotFoundError(msg)
        return path

    def _get_scratch_dir(self, task: Task) -> Path:
        """Return stable scratch directory for a task.

        Args:
            task: Task requesting a scratch directory.

        Returns:
            Per-task directory path keyed by resolved hash.
        """
        key_str = task.resolved_hash(workspace=self).b32()
        d = self._directory_path / "scratch" / key_str[:2] / f"{key_str}"
        d.mkdir(parents=True, exist_ok=True)
        logger.debug("Resolved scratch dir for task %s: %s.", task, d)
        return d

    def _task_log_dir(self, task: Task) -> tuple[Path, str]:
        """Sharded task-log directory behind the base path-backed log methods."""
        key_str = task.resolved_hash(workspace=self).b32()
        log_dir = self._directory_path / "task_logs" / key_str[:2]
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir, key_str
