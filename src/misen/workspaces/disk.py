"""Disk-backed workspace implementation.

This backend persists:

- task/resolved/result hash indices in LMDB
- result payload directories on disk
- lock files for cross-process coordination (NFS-compatible)
- task/job logs for runtime observability

The design prioritizes deterministic paths, write-once result materialization,
and lock-based safety for concurrent producers.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Generator, Iterator, MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Literal, Self, TextIO, TypeVar, cast

import lmdb

from misen.utils.hashing import Hash, ResolvedTaskHash, ResultHash, TaskHash
from misen.utils.locks import LockLike, NFSLock
from misen.workspace import Workspace

if TYPE_CHECKING:
    from misen.tasks import Task

KT = TypeVar("KT", bound=Hash)
VT = TypeVar("VT", bound=Hash)
logger = logging.getLogger(__name__)


class LMDBMapping(MutableMapping[KT, VT], Generic[KT, VT]):
    """Typed key/value mapping backed by a single LMDB database.

    Keys and values must be :class:`misen.utils.hashing.Hash` subclasses that
    support ``encode``/``decode``.
    """

    _key_type: type[KT]
    _value_type: type[VT]
    lock: NFSLock
    env: lmdb.Environment
    _closed: bool
    _path: str

    __slots__ = ("_closed", "_key_type", "_path", "_value_type", "env", "lock")

    def __class_getitem__(cls, item: tuple[type[KT], type[VT]]) -> type[Self]:
        """Parameterize mapping with concrete key/value hash types.

        Args:
            item: ``(KeyHashType, ValueHashType)`` tuple.

        Returns:
            Specialized ``LMDBMapping`` subclass.
        """
        key_t, val_t = item
        return cast(
            "type[Self]",
            type(
                f"{cls.__name__}[{key_t.__name__},{val_t.__name__}]",
                (cls,),
                {"_key_type": key_t, "_value_type": val_t, "__module__": cls.__module__},
            ),
        )

    def __init__(self, database_path: Path) -> None:
        """Initialize the LMDB database mapping."""
        if not hasattr(self, "_key_type") or not hasattr(self, "_value_type"):
            msg = "Construct as LMDBMapping[KeyType, ValueType](...)"
            raise TypeError(msg)

        self.lock = NFSLock(database_path.with_suffix(".lock"), lifetime=10)

        self._path = str(database_path)
        self.env = lmdb.Environment(
            self._path,
            subdir=False,
            lock=False,
            map_size=2**28,  # 256 MiB
            max_dbs=1,
        )
        self._closed = False

    def _check_open(self) -> None:
        """Raise if the underlying environment has been closed."""
        if self._closed:
            msg = f"LMDBMapping at {self._path!r} is closed."
            raise RuntimeError(msg)

    def close(self) -> None:
        """Close the LMDB environment. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self.env.close()

    def __len__(self) -> int:
        """Return number of entries in the database."""
        self._check_open()
        return self.env.stat()["entries"]

    def __iter__(self) -> Generator[KT]:
        """Iterate over typed keys in the database."""
        self._check_open()
        with self.env.begin() as txn:
            for k, _ in txn.cursor():
                yield self._key_type.decode(k)

    def __contains__(self, key: object) -> bool:
        """Return whether key exists.

        Args:
            key: Candidate key.

        Returns:
            ``True`` if key is of expected type and present.
        """
        if not isinstance(key, self._key_type):
            return False
        self._check_open()
        with self.env.begin() as txn:
            return txn.get(key.encode()) is not None

    def __getitem__(self, key: KT) -> VT:
        """Return the value for the given key.

        Raises:
            KeyError: If the key is not present.
        """
        self._check_open()
        with self.env.begin() as txn:
            v: bytes | None = txn.get(key.encode(), default=None)
            if v is None:
                raise KeyError(key)
            return self._value_type.decode(v)

    def __setitem__(self, key: KT, value: VT) -> None:
        """Store key/value pair atomically.

        Args:
            key: Hash key.
            value: Hash value.
        """
        self._check_open()
        _key, _value = key.encode(), value.encode()
        with self.lock.context(blocking=True):
            with self.env.begin(write=True) as txn:
                txn.put(_key, _value)

    def __delitem__(self, key: KT) -> None:
        """Remove a key/value pair.

        Raises:
            KeyError: If the key is not present.
        """
        self._check_open()
        _key = key.encode()
        with self.lock.context(blocking=True):
            with self.env.begin(write=True) as txn:
                success = txn.delete(_key)
        if not success:
            raise KeyError(key)

    def clear(self) -> None:
        """Delete all entries from the database."""
        self._check_open()
        with self.lock.context(blocking=True):
            with self.env.begin(write=True) as txn:
                for _k, _ in txn.cursor():
                    txn.delete(_k)


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
        return isinstance(key, ResultHash) and self._result_dir_path(key).exists()

    def __getitem__(self, key: ResultHash) -> Path:
        """Return the directory for a result hash.

        Raises:
            KeyError: If the result directory is missing.
        """
        result_dir_path = self._result_dir_path(key)
        if not result_dir_path.exists():
            raise KeyError(key)
        return result_dir_path

    def __setitem__(self, key: ResultHash, value: Path) -> None:
        """Persist result directory if key is not already present.

        Args:
            key: Result hash.
            value: Temporary directory containing serialized payload.
        """
        result_dir_path = self._result_dir_path(key)
        if not result_dir_path.exists():
            result_dir_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(value, result_dir_path)
            self._fsync_dir(result_dir_path.parent)

    def __delitem__(self, key: ResultHash) -> None:
        """Delete a result directory.

        Raises:
            KeyError: If the directory is missing.
        """
        result_dir_path = self._result_dir_path(key)
        if not result_dir_path.exists():
            raise KeyError(key)
        # atomic deletion
        trash_dir = Path(
            tempfile.mkdtemp(dir=result_dir_path.parent, prefix=f"{result_dir_path.name}.", suffix=".trash")
        )
        shutil.move(result_dir_path, trash_dir)
        self._fsync_dir(result_dir_path.parent)
        shutil.rmtree(trash_dir)
        self._fsync_dir(trash_dir.parent)

    def __iter__(self) -> Iterator[ResultHash]:
        """Iterate over stored result hashes."""
        for p in self.directory.glob(("[0-9a-f]" * 2) + "/" + ("[0-9a-f]" * 16)):
            if (stem := p.stem)[:2] == p.parent.name:
                yield ResultHash(stem, base=16)

    def __len__(self) -> int:
        """Return number of stored results."""
        return sum(1 for _ in self)

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        """Fsync a directory descriptor.

        Args:
            path: Directory path.
        """
        fd = os.open(path, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class DiskWorkspace(Workspace):
    """Workspace implementation backed by local/NFS-accessible directories."""

    directory: str = ".misen"

    def __post_init__(self) -> None:
        """Create directory layout and initialize persistent caches."""
        self._directory = Path(self.directory)
        self._directory.mkdir(exist_ok=True)
        self.get_temp_dir().mkdir(parents=True, exist_ok=True)
        (self._directory / "work").mkdir(parents=True, exist_ok=True)
        (self._directory / "task_logs").mkdir(parents=True, exist_ok=True)
        (self.get_temp_dir() / "job_logs").mkdir(parents=True, exist_ok=True)
        (self.get_temp_dir() / "task_locks").mkdir(parents=True, exist_ok=True)
        (self.get_temp_dir() / "result_locks").mkdir(parents=True, exist_ok=True)

        super()._post_init(
            resolved_hash_cache=LMDBMapping[TaskHash, ResolvedTaskHash](self._directory / "resolved_hash_cache.mdb"),
            result_hash_cache=LMDBMapping[ResolvedTaskHash, ResultHash](self._directory / "result_hash_cache.mdb"),
            result_store=DiskResultStore(self._directory / "results"),
        )
        logger.info("Initialized DiskWorkspace at %s.", self._directory.resolve())

    def close(self) -> None:
        """Release LMDB environments backing the workspace caches.

        Idempotent. Subsequent operations against the closed caches raise
        ``RuntimeError``. Use ``contextlib.closing(workspace)`` for
        scoped cleanup. Note that workspace instances are memoized by
        constructor kwargs, so re-constructing the same workspace after
        close returns the same (closed) instance within the process.
        """
        for cache in (self._resolved_hash_cache, self._result_hash_cache):
            if isinstance(cache, LMDBMapping):
                cache.close()

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
        return Path(self._directory) / "tmp"

    def _get_work_dir(self, task: Task) -> Path:
        """Return stable working directory for a task.

        Args:
            task: Task requesting a work directory.

        Returns:
            Per-task directory path keyed by resolved hash.
        """
        key_str = task.resolved_hash(workspace=self).b32()
        d = Path(self._directory) / "work" / key_str[:2] / f"{key_str}"
        d.mkdir(parents=True, exist_ok=True)
        logger.debug("Resolved work dir for task %s: %s.", task, d)
        return d

    def open_task_log(
        self,
        task: Task,
        mode: Literal["a", "r"],
        job_id: str | None = None,
    ) -> TextIO:
        """Open a task log file in the workspace.

        Args:
            task: Task whose logs to open.
            mode: File mode (``"a"`` or ``"r"``).
            job_id: Optional job identifier to filter/select log stream.
                In read mode with no job_id, opens the most recently modified log.

        Returns:
            Open text file object.

        Raises:
            FileNotFoundError: If ``mode="r"`` and no matching logs exist.
        """
        key_str = task.resolved_hash(workspace=self).b32()
        log_dir = Path(self._directory) / "task_logs" / key_str[:2]
        log_dir.mkdir(parents=True, exist_ok=True)

        if job_id is None and mode == "r":
            # Find the most recently modified log for this task.
            matches = sorted(log_dir.glob(f"{key_str}_*.log"), key=lambda p: p.stat().st_mtime)
            if not matches:
                msg = f"No logs found for {key_str} in {log_dir}"
                raise FileNotFoundError(msg)
            log_path = matches[-1]
        else:
            selected_job_id = job_id or "0"
            log_path = log_dir / f"{key_str}_{selected_job_id}.log"

        logger.debug(
            "Opening task log for %s at %s (mode=%s, job_id=%s).",
            task,
            log_path,
            mode,
            job_id,
        )
        return log_path.open(mode, buffering=1)
