"""Workspace interfaces for caching task results."""

from __future__ import annotations

import contextlib
import shutil
from abc import abstractmethod
from collections.abc import Iterator, MutableMapping
from typing import TYPE_CHECKING, Any, Literal, TextIO, TypeAlias, TypeVar, cast, get_args

from typing_extensions import assert_never

from misen.task import Task
from misen.utils.settings import FromSettingsABC

if TYPE_CHECKING:
    from pathlib import Path

    from misen.utils.hashes import ResolvedTaskHash, ResultHash, TaskHash
    from misen.utils.locks import LockLike

__all__ = ["Workspace"]

R = TypeVar("R")


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

    @property
    def results(self) -> ResultMap:
        """Return the result map interface for cached task results."""
        return self._result_map

    @abstractmethod
    def lock(self, namespace: Literal["task", "result"], key: str) -> LockLike:
        """Return a lock for task or result namespaces."""
        ...

    @abstractmethod
    def get_temp_dir(self) -> Path:
        """Return a temporary directory for workspace operations."""
        ...

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
        ...

    @abstractmethod
    def get_job_log_path(self, job_id: str) -> Path:
        """Return the path to an executor job log file."""
        ...


class ResultMap(MutableMapping[Task[Any], Any]):
    """Mapping interface for task results stored in a workspace."""

    __slots__ = ("result_store", "workspace")

    def __init__(self, result_store: MutableMapping[ResultHash, Path], workspace: Workspace) -> None:
        """Initialize the result map wrapper."""
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
        return key.properties.serializer.load(directory)

    def __setitem__(self, key: Task[R], value: R, /) -> None:
        """Persist a result for the given task."""
        result_hash = key.result_hash(workspace=self.workspace)
        with self.workspace.lock(namespace="result", key=result_hash.b32()).context(blocking=True, timeout=None):
            if result_hash not in self.result_store:
                tmp_dir = self.workspace.get_temp_dir() / "results" / result_hash.b32()
                tmp_dir.mkdir(parents=True, exist_ok=True)
                key.properties.serializer.save(value, tmp_dir)
                self.result_store[result_hash] = tmp_dir
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
        except Exception as e:
            msg = f"Result for task {key} not found in cache."
            raise KeyError(msg) from e

    def __iter__(self) -> Iterator[Task]:
        """Iterate over tasks in the mapping."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of cached results."""
        return len(self.result_store)

    def __contains__(self, key: object, /) -> bool:
        """Return True if the task has a cached result."""
        if not isinstance(key, Task):
            return False
        try:
            result_hash = key.result_hash(workspace=self.workspace)
        except RuntimeError:
            return False
        return result_hash in self.result_store
