from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Literal,
    TypeAlias,
    TypeVar,
)

from misen.utils.settings import FromSettingsABC

from .task import Task

if TYPE_CHECKING:
    from pathlib import Path

    from .utils.hashes import ResolvedTaskHash, ResultHash, TaskHash
    from .utils.locks import LockLike

__all__ = ["Workspace"]

R = TypeVar("R")


WorkspaceType: TypeAlias = Literal["disk"]


class Workspace(FromSettingsABC):
    def __post_init__(
        self,
        resolved_hash_cache: MutableMapping[TaskHash, ResolvedTaskHash],
        result_hash_cache: MutableMapping[ResolvedTaskHash, ResultHash],
        result_store: MutableMapping[ResultHash, Path],
        log_store: MutableMapping[ResolvedTaskHash, TaskLogs],
    ):
        # session-only (non-persistent) caches
        self._resolved_hashes: dict[TaskHash, ResolvedTaskHash] = {}
        self._result_hashes: dict[TaskHash, ResultHash] = {}

        # workspace caches
        self._resolved_hash_cache = resolved_hash_cache
        self._result_hash_cache = result_hash_cache
        self._result_store = result_store
        self._log_store = log_store

        # public accessors to workspace data
        self.results = ResultMap(workspace=self)
        self.logs = LogMap(workspace=self)

    @abstractmethod
    def lock(self, namespace: Literal["task", "result"], key: str) -> LockLike: ...

    @abstractmethod
    def get_temp_dir(self) -> Path: ...

    @abstractmethod
    def get_work_dir(self, task: Task) -> Path:
        """Return a directory where the task can store working files. E.g. to cache intermediate results."""
        ...

    @classmethod
    def _settings_key(cls) -> str:
        return "workspace"

    @classmethod
    def _default_type_name(cls) -> str:
        return "disk"

    @classmethod
    def _default_kwargs(cls) -> dict:
        return {"directory": ".misen"}

    @classmethod
    def _resolve_type(cls, type_name: str | WorkspaceType) -> type["Workspace"]:
        match type_name:
            case "disk":
                from misen.workspaces.disk import DiskWorkspace

                return DiskWorkspace
        return super()._resolve_type(type_name)


class ResultMap(MutableMapping[Task[Any], Any]):
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def __getitem__(self, key: Task[R], /) -> R:
        try:
            result_hash = key._result_hash(workspace=self.workspace)
            dir = self.workspace._result_store[result_hash]
        except Exception as e:
            raise KeyError(f"Result for task {key} not found in cache.") from e
        return key.properties.serializer.load(dir)

    def __setitem__(self, key: Task[R], value: R, /) -> None:
        result_hash = key._result_hash(workspace=self.workspace)
        with self.workspace.lock(namespace="result", key=result_hash.hex()).context(blocking=True, timeout=None):
            if result_hash not in self.workspace._result_store:
                tmp_dir = self.workspace.get_temp_dir() / "results" / result_hash.hex()
                tmp_dir.mkdir(parents=True, exist_ok=True)
                key.properties.serializer.save(value, tmp_dir)
                self.workspace._result_store[result_hash] = tmp_dir
                try:
                    shutil.rmtree(tmp_dir)
                except FileNotFoundError:
                    pass

    def __delitem__(self, key: Task[R], /) -> None:
        try:
            result_hash = key._result_hash(workspace=self.workspace)
            del self.workspace._result_store[result_hash]
        except Exception as e:
            raise KeyError(f"Result for task {key} not found in cache.") from e

    def __iter__(self) -> Iterator[Task]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.workspace._result_store)

    def __contains__(self, key: object, /) -> bool:
        if not isinstance(key, Task):
            return False
        try:
            result_hash = key._result_hash(workspace=self.workspace)
        except RuntimeError:
            return False
        return result_hash in self.workspace._result_store


class LogMap(MutableMapping[Task, "TaskLogs"]):
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def __getitem__(self, key: Task, /) -> TaskLogs:
        resolved_hash: ResolvedTaskHash = key._resolved_hash(workspace=self.workspace)
        return self.workspace._log_store[resolved_hash]

    def __setitem__(self, key: Task, value: TaskLogs, /) -> None:
        resolved_hash: ResolvedTaskHash = key._resolved_hash(workspace=self.workspace)
        self.workspace._log_store[resolved_hash] = value

    def __delitem__(self, key: Task, /) -> None:
        resolved_hash: ResolvedTaskHash = key._resolved_hash(workspace=self.workspace)
        del self.workspace._log_store[resolved_hash]

    def __iter__(self) -> Iterator[Task]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.workspace._log_store)

    def __contains__(self, key: object, /) -> bool:
        if not isinstance(key, Task):
            return False
        resolved_hash: ResolvedTaskHash = key._resolved_hash(workspace=self.workspace)
        return resolved_hash in self.workspace._log_store


# TODO: append, read, support multiple runs
class TaskLogs(ABC):
    pass
