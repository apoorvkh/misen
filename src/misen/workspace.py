"""Workspace abstraction for caching, locking, and runtime artifacts.

``Workspace`` isolates storage concerns from execution concerns:

- Executors schedule and run work.
- Tasks describe computation and identity.
- Workspace persists hashes/results and coordinates cross-process locks.
- Runtime lock contract: for cacheable tasks, a workspace lock keyed by
  resolved task identity enforces at most one active execution at a time.
  Non-cacheable tasks are not serialized by this runtime lock.

This separation keeps execution backends modular while preserving one
consistent cache/locking contract.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
from abc import abstractmethod
from collections.abc import Iterator, MutableMapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TextIO, TypeAlias, TypeVar

from misen.exceptions import CacheError
from misen.tasks import Task
from misen.utils import serde
from misen.utils.settings import Configurable

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from misen.utils.hashing import ResolvedTaskHash, ResultHash, TaskHash
    from misen.utils.locks import LockLike
    from misen.utils.work_unit import WorkUnit

__all__ = ["Workspace"]


WorkspaceType: TypeAlias = Literal["disk"]
TRACE_LEVEL = logging.DEBUG - 5
logger = logging.getLogger(__name__)


class Workspace(Configurable):
    """Base class for workspace storage backends.

    Concrete implementations provide persistence (for hashes/results), lock
    implementations, and task/job log storage.
    """

    _config_key: ClassVar[str] = "workspace"
    _config_default_type: ClassVar[str] = "misen.workspaces.disk:DiskWorkspace"
    _config_aliases: ClassVar[dict[WorkspaceType, str]] = {"disk": "misen.workspaces.disk:DiskWorkspace"}

    def _post_init(
        self,
        resolved_hash_cache: MutableMapping[TaskHash, ResolvedTaskHash],
        result_hash_cache: MutableMapping[ResolvedTaskHash, ResultHash],
        result_store: MutableMapping[ResultHash, Path],
    ) -> None:
        """Initialize workspace caches and storage backends.

        Args:
            resolved_hash_cache: Persistent cache for task resolved hashes.
            result_hash_cache: Persistent cache for task result hashes.
            result_store: Store mapping result hashes to on-disk directories.
        """
        # Session-local hot caches reduce repeated backend lookups.
        self._resolved_hashes: dict[TaskHash, ResolvedTaskHash] = {}
        self._result_hashes: dict[TaskHash, ResultHash] = {}

        # Persistent caches/stores are backend-specific (e.g., LMDB + disk).
        self._resolved_hash_cache: MutableMapping[TaskHash, ResolvedTaskHash] = resolved_hash_cache
        self._result_hash_cache: MutableMapping[ResolvedTaskHash, ResultHash] = result_hash_cache
        self._result_map = ResultMap(result_store=result_store, workspace=self)

    def get_resolved_hash(self, task: Task) -> ResolvedTaskHash | None:
        """Return cached resolved hash for a task, if available.

        Args:
            task: Task to query.

        Returns:
            Resolved hash if present, otherwise ``None``.
        """
        task_hash = task.task_hash()
        # Fast path: in-memory session cache.
        resolved_hash = self._resolved_hashes.get(task_hash)
        if resolved_hash is not None:
            logger.log(TRACE_LEVEL, "Resolved-hash memory cache hit for task %s.", task)
            return resolved_hash
        # Slow path: persistent workspace cache.
        resolved_hash = self._resolved_hash_cache.get(task_hash)
        # Promote to session cache after a persistent hit.
        if resolved_hash is not None:
            self._resolved_hashes[task_hash] = resolved_hash
            logger.log(TRACE_LEVEL, "Resolved-hash persistent cache hit for task %s.", task)
        else:
            logger.log(TRACE_LEVEL, "Resolved-hash cache miss for task %s.", task)
        return resolved_hash

    def set_resolved_hash(self, task: Task, resolved_hash: ResolvedTaskHash) -> None:
        """Persist resolved hash for a task.

        Args:
            task: Task to update.
            resolved_hash: Resolved task hash value.
        """
        task_hash = task.task_hash()
        self._resolved_hashes[task_hash] = resolved_hash
        self._resolved_hash_cache[task_hash] = resolved_hash
        logger.debug("Stored resolved hash for task %s.", task)

    def get_result_hash(self, task: Task) -> ResultHash:
        """Return the result hash for a completed task.

        Raises:
            CacheError: If the task has not been computed yet.
        """
        # Fast path: in-memory session cache.
        task_hash = task.task_hash()
        result_hash = self._result_hashes.get(task_hash)
        if result_hash is not None:
            logger.log(TRACE_LEVEL, "Result-hash memory cache hit for task %s.", task)
            return result_hash
        # Slow path: persistent workspace cache by resolved task identity.
        resolved_hash = task.resolved_hash(workspace=self)
        result_hash = self._result_hash_cache.get(resolved_hash)
        if result_hash is None:
            logger.log(TRACE_LEVEL, "Result-hash cache miss for task %s.", task)
            msg = f"Task {task} must be computed first."
            raise CacheError(msg)
        # Promote to session cache after a persistent hit.
        self._result_hashes[task_hash] = result_hash
        logger.log(TRACE_LEVEL, "Result-hash persistent cache hit for task %s.", task)
        return result_hash

    def set_result_hash(self, task: Task, result_hash: ResultHash) -> None:
        """Persist result hash for a task.

        Args:
            task: Task to update.
            result_hash: Result hash value.
        """
        self._result_hashes[task.task_hash()] = result_hash
        resolved_hash = task.resolved_hash(workspace=self)
        self._result_hash_cache[resolved_hash] = result_hash
        logger.debug("Stored result hash for task %s.", task)

    def clear_result_hash(self, task: Task) -> None:
        """Remove persisted result-hash mapping for a task.

        Args:
            task: Task whose mapping should be removed.
        """
        del self._result_hashes[task.task_hash()]
        resolved_hash = task.resolved_hash(workspace=self)
        del self._result_hash_cache[resolved_hash]
        logger.debug("Cleared result hash for task %s.", task)

    @property
    def results(self) -> ResultMap:
        """Return mapping-like interface for cached task results."""
        return self._result_map

    @abstractmethod
    def lock(self, namespace: Literal["task", "result"], key: str) -> LockLike:
        """Return a lock object for task/result namespaces.

        Args:
            namespace: Lock namespace (task runtime or result materialization).
            key: Lock key unique within the namespace.

        Returns:
            Lock-like object with acquire/release/context APIs.

        Notes:
            ``namespace="task"`` is used for cacheable task runtime exclusion
            (single active execution per workspace/key). ``namespace="result"``
            is used for serialized result materialization.
        """

    @abstractmethod
    def get_temp_dir(self) -> Path:
        """Return temporary directory used for workspace operations."""

    def get_work_dir(self, task: Task) -> Path:
        """Return a per-task working directory for cacheable tasks.

        Args:
            task: Task requesting its working directory.

        Returns:
            Filesystem path for runtime intermediate artifacts.

        Raises:
            RuntimeError: If the task is non-cacheable.
        """
        if not task.properties.cache:
            msg = f"{task} cannot use workspace work_dir unless Task.properties.cache == True."
            raise RuntimeError(msg)
        return self._get_work_dir(task)

    @abstractmethod
    def _get_work_dir(self, task: Task) -> Path: ...

    @abstractmethod
    def open_task_log(
        self,
        task: Task,
        mode: Literal["a", "r"],
        job_id: str | None = None,
    ) -> TextIO:
        """Open a log file for the given task.

        Logs are keyed by ``(resolved_hash, job_id)``. Each task execution
        produces one log file per job.

        Implementations must support:

        - **Write mode** (``mode="a"``): open (or create) the log for the
          given task and job_id. If ``job_id`` is ``None``, use a default
          identifier.
        - **Read mode** (``mode="r"``): if ``job_id`` is provided, open that
          specific log. If ``job_id`` is ``None``, open the most recent log
          for the task. Recency is implementation-defined (e.g., filesystem
          mtime, object-store upload timestamp, or an internal index).
        - Raise ``FileNotFoundError`` in read mode when no matching log exists.

        Args:
            task: Task associated with the log file.
            mode: File open mode (``"a"`` for write/append, ``"r"`` for read).
            job_id: Optional job identifier. Selects a specific log in read
                mode; groups output in write mode.
        """

    def get_job_log(self, job_id: str, work_unit: WorkUnit) -> Path:
        """Return job-log path for a work unit.

        Args:
            job_id: Backend job identifier.
            work_unit: Work unit associated with the job.

        Returns:
            Path where the backend should write combined job logs.
        """
        log_dir = self.get_temp_dir() / "job_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        work_unit_prefix = work_unit.root.task_hash().b32()
        path = log_dir / f"{work_unit_prefix}_{job_id}.log"
        logger.debug("Resolved job log path for work unit %s: %s.", work_unit, path)
        return path

    def job_log_iter(self, work_unit: WorkUnit | None = None) -> Iterator[Path]:
        """Return iterator over job-log files.

        Args:
            work_unit: Optional filter for a specific work unit.

        Returns:
            Iterator of log-file paths.
        """
        log_dir = self.get_temp_dir() / "job_logs"
        if work_unit is None:
            logger.debug("Iterating all job logs in %s.", log_dir)
            return log_dir.iterdir()

        work_unit_prefix = work_unit.root.task_hash().b32()
        logger.debug("Iterating job logs in %s for work unit %s.", log_dir, work_unit)
        return log_dir.glob(f"{work_unit_prefix}_*.log")


R = TypeVar("R")


class ResultMap(MutableMapping[Task[Any], Any]):
    """Mapping-like interface over workspace result storage.

    Keys are :class:`misen.tasks.Task` objects and values are deserialized
    result payloads.
    """

    __slots__ = ("result_store", "workspace")

    def __init__(self, result_store: MutableMapping[ResultHash, Path], workspace: Workspace) -> None:
        """Initialize result-map wrapper.

        Args:
            result_store: Mapping from result hash to payload directory.
            workspace: Owning workspace.
        """
        self.result_store = result_store
        self.workspace = workspace

    def __getitem__(self, key: Task[R], /) -> R:
        """Return the cached result for a task.

        Raises:
            KeyError: If the result is not present in the cache.
        """
        try:
            result_hash = key.result_hash(workspace=self.workspace)
            directory = self.result_store[result_hash]
        except Exception as e:
            msg = f"Result for task {key} not found in cache."
            raise KeyError(msg) from e
        logger.debug("Loading cached result for task %s from %s.", key, directory)
        return serde.load(directory, ser_cls=key.properties.serializer)

    def __setitem__(self, key: Task[R], value: R, /) -> None:
        """Persist result for the given task.

        Args:
            key: Task key.
            value: Computed result value.
        """
        result_hash = key.result_hash(workspace=self.workspace)
        with self.workspace.lock(namespace="result", key=result_hash.b32()).context(blocking=True, timeout=None):
            if result_hash in self.result_store:
                logger.debug("Result store already has payload for task %s.", key)
                return
            tmp_dir = self.workspace.get_temp_dir() / "results" / result_hash.b32()
            tmp_dir.mkdir(parents=True, exist_ok=True)
            serde.save(value, tmp_dir, ser_cls=key.properties.serializer)
            self.result_store[result_hash] = tmp_dir
            logger.debug("Stored cached result for task %s at %s.", key, tmp_dir)
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(tmp_dir)

    def __delitem__(self, key: Task[R], /) -> None:
        """Remove a cached result for a task.

        Raises:
            KeyError: If the result is not present in the cache.
        """
        try:
            result_hash = key.result_hash(workspace=self.workspace)
            del self.result_store[result_hash]
            logger.debug("Deleted cached result payload for task %s.", key)
        except Exception as e:
            msg = f"Result for task {key} not found in cache."
            raise KeyError(msg) from e

    def __iter__(self) -> Iterator[Task]:
        """Iterate over task keys.

        Notes:
            This is not implemented because the persistent store is keyed by
            result hashes rather than by task identity.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return number of cached results."""
        return len(self.result_store)

    def __contains__(self, key: object, /) -> bool:
        """Return True if the task has a cached result."""
        if not isinstance(key, Task):
            return False
        try:
            result_hash = key.result_hash(workspace=self.workspace)
        except CacheError:
            return False
        return result_hash in self.result_store
