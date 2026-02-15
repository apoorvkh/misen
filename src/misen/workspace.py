"""Workspace interfaces for caching task results."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal, TextIO, TypeAlias, cast, get_args

from typing_extensions import assert_never

from misen.utils.settings import FromSettingsABC
from misen.utils.workspace_result_map import ResultMap

if TYPE_CHECKING:
    from collections.abc import Iterator, MutableMapping
    from pathlib import Path

    from misen.task import Task
    from misen.utils.hashes import ResolvedTaskHash, ResultHash, TaskHash
    from misen.utils.locks import LockLike
    from misen.utils.work_unit import WorkUnit

__all__ = ["Workspace"]


WorkspaceType: TypeAlias = Literal["disk"]


class Workspace(FromSettingsABC):
    """Base class for workspace storage backends."""

    @staticmethod
    def _settings_key() -> str:
        """Return the TOML settings key for workspace configuration."""
        return "workspace"

    @staticmethod
    def _default() -> Workspace:
        """Return the default workspace implementation."""
        from misen.workspaces.disk import DiskWorkspace

        return DiskWorkspace()

    @classmethod
    def _resolve_type(cls, type_name: str | WorkspaceType) -> type[Workspace]:
        """Resolve a workspace type name to a class."""
        if type_name in get_args(WorkspaceType):
            type_name = cast("WorkspaceType", type_name)
            match type_name:
                case "disk":
                    from misen.workspaces.disk import DiskWorkspace

                    return DiskWorkspace
                case _:
                    assert_never(type_name)
        return super()._resolve_type(type_name)

    def __post_init__(
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
        # session-only (non-persistent) caches
        self._resolved_hashes: dict[TaskHash, ResolvedTaskHash] = {}
        self._result_hashes: dict[TaskHash, ResultHash] = {}

        # workspace caches
        self._resolved_hash_cache: MutableMapping[TaskHash, ResolvedTaskHash] = resolved_hash_cache
        self._result_hash_cache: MutableMapping[ResolvedTaskHash, ResultHash] = result_hash_cache
        self._result_map = ResultMap(result_store=result_store, workspace=self)

    def get_resolved_hash(self, task: Task) -> ResolvedTaskHash | None:
        """Return the resolved hash for a task if cached."""
        task_hash = task.task_hash()
        # fast (session-only) cache
        resolved_hash = self._resolved_hashes.get(task_hash)
        if resolved_hash is not None:
            return resolved_hash
        # slower (persistent) cache
        resolved_hash = self._resolved_hash_cache.get(task_hash)
        # update fast cache if possible
        if resolved_hash is not None:
            self._resolved_hashes[task_hash] = resolved_hash
        return resolved_hash

    def set_resolved_hash(self, task: Task, resolved_hash: ResolvedTaskHash) -> None:
        """Persist the resolved hash for a task."""
        task_hash = task.task_hash()
        self._resolved_hashes[task_hash] = resolved_hash
        self._resolved_hash_cache[task_hash] = resolved_hash

    def get_result_hash(self, task: Task) -> ResultHash:
        """Return the result hash for a completed task.

        Raises:
            RuntimeError: If the task has not been computed yet.
        """
        # fast (session-only) cache
        task_hash = task.task_hash()
        result_hash = self._result_hashes.get(task_hash)
        if result_hash is not None:
            return result_hash
        # slower (persistent) cache
        resolved_hash = task.resolved_hash(workspace=self)
        result_hash = self._result_hash_cache.get(resolved_hash)
        if result_hash is None:
            msg = f"Task {task} must be computed first."
            raise RuntimeError(msg)
        # update fast cache
        self._result_hashes[task_hash] = result_hash
        return result_hash

    def set_result_hash(self, task: Task, result_hash: ResultHash) -> None:
        """Persist the result hash for a task."""
        self._result_hashes[task.task_hash()] = result_hash
        resolved_hash = task.resolved_hash(workspace=self)
        self._result_hash_cache[resolved_hash] = result_hash

    def clear_result_hash(self, task: Task) -> None:
        """Remove the result hash for a task."""
        del self._result_hashes[task.task_hash()]
        resolved_hash = task.resolved_hash(workspace=self)
        del self._result_hash_cache[resolved_hash]

    @property
    def results(self) -> ResultMap:
        """Return the result map interface for cached task results."""
        return self._result_map

    @abstractmethod
    def lock(self, namespace: Literal["task", "result"], key: str) -> LockLike:
        """Return a lock for task or result namespaces."""

    @abstractmethod
    def get_temp_dir(self) -> Path:
        """Return a temporary directory for workspace operations."""

    @abstractmethod
    def get_work_dir(self, task: Task) -> Path:
        """Return a working directory for cacheable tasks. E.g. to cache intermediate results during task runtime."""
        if not task.properties.cache:
            msg = f"{task} cannot use workspace work_dir unless Task.properties.cache == True."
            raise RuntimeError(msg)

    @abstractmethod
    def open_task_log(
        self,
        task: Task,
        mode: Literal["a", "r"],
        job_id: str | None = None,
        timestamp: int | Literal["current", "latest"] = "latest",
    ) -> TextIO:
        """Open a log file for the given task.

        Args:
            task: Task associated with the log file.
            mode: File open mode, usually "a" or "r".
            timestamp: Timestamp to select a log file, or "latest".
            job_id: Optional job identifier to group task logs from the same executor job.
        """

    def get_job_log(self, job_id: str, work_unit: WorkUnit) -> Path:
        """Return a path for a job's logs."""
        log_dir = self.get_temp_dir() / "job_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        work_unit_prefix = work_unit.root.task_hash().b32()
        return log_dir / f"{work_unit_prefix}_{job_id}.log"

    def job_log_iter(self, work_unit: WorkUnit | None = None) -> Iterator[Path]:
        """Return an iterator over job log paths corresonding to given WorkUnit."""
        log_dir = self.get_temp_dir() / "job_logs"
        if work_unit is None:
            return log_dir.iterdir()

        work_unit_prefix = work_unit.root.task_hash().b32()
        return (log_dir / f"{work_unit_prefix}_*.log").iterdir()
