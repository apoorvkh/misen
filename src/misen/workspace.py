from __future__ import annotations

import contextlib
import shutil
from abc import abstractmethod
from collections.abc import Iterator, MutableMapping
from typing import TYPE_CHECKING, Any, Literal, TextIO, TypeAlias, TypeVar, cast, get_args

from typing_extensions import assert_never

from .task import Task
from .utils.settings import FromSettingsABC

if TYPE_CHECKING:
    from pathlib import Path

    from .utils.hashes import ResolvedTaskHash, ResultHash, TaskHash
    from .utils.locks import LockLike

__all__ = ["Workspace"]

R = TypeVar("R")


WorkspaceType: TypeAlias = Literal["disk"]


class Workspace(FromSettingsABC):
    @staticmethod
    def _settings_key() -> str:
        return "workspace"

    @staticmethod
    def _default() -> Workspace:
        from .workspaces.disk import DiskWorkspace

        return DiskWorkspace()

    @classmethod
    def _resolve_type(cls, type_name: str | WorkspaceType) -> type[Workspace]:
        if type_name in get_args(WorkspaceType):
            type_name = cast("WorkspaceType", type_name)
            match type_name:
                case "disk":
                    from .workspaces.disk import DiskWorkspace

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
        # session-only (non-persistent) caches
        self._resolved_hashes: dict[TaskHash, ResolvedTaskHash] = {}
        self._result_hashes: dict[TaskHash, ResultHash] = {}

        # workspace caches
        self._resolved_hash_cache: MutableMapping[TaskHash, ResolvedTaskHash] = resolved_hash_cache
        self._result_hash_cache: MutableMapping[ResolvedTaskHash, ResultHash] = result_hash_cache
        self._result_map = ResultMap(result_store=result_store, workspace=self)

    def get_resolved_hash(self, task: Task) -> ResolvedTaskHash | None:
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
        task_hash = task.task_hash()
        self._resolved_hashes[task_hash] = resolved_hash
        self._resolved_hash_cache[task_hash] = resolved_hash

    def get_result_hash(self, task: Task) -> ResultHash:
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
        self._result_hashes[task.task_hash()] = result_hash
        resolved_hash = task.resolved_hash(workspace=self)
        self._result_hash_cache[resolved_hash] = result_hash

    @property
    def results(self) -> ResultMap:
        return self._result_map

    @abstractmethod
    def lock(self, namespace: Literal["task", "result"], key: str) -> LockLike: ...

    @abstractmethod
    def get_temp_dir(self) -> Path: ...

    @abstractmethod
    def get_work_dir(self, task: Task) -> Path:
        """Return a directory where the task can store working files. E.g. to cache intermediate results."""
        ...

    ## TODO: non-unique (maybe use hostname, pid, ?)
    @abstractmethod
    def open_log(
        self,
        task: Task,
        mode: Literal["a", "r"],
        timestamp: int | Literal["latest"] = "latest",
    ) -> TextIO: ...


class ResultMap(MutableMapping[Task[Any], Any]):
    def __init__(self, result_store: MutableMapping[ResultHash, Path], workspace: Workspace) -> None:
        self.result_store = result_store
        self.workspace = workspace

    def __getitem__(self, key: Task[R], /) -> R:
        try:
            result_hash = key.result_hash(workspace=self.workspace)
            directory = self.result_store[result_hash]
        except Exception as e:
            msg = f"Result for task {key} not found in cache."
            raise KeyError(msg) from e
        return key.properties.serializer.load(directory)

    def __setitem__(self, key: Task[R], value: R, /) -> None:
        result_hash = key.result_hash(workspace=self.workspace)
        with self.workspace.lock(namespace="result", key=result_hash.hex()).context(blocking=True, timeout=None):
            if result_hash not in self.result_store:
                tmp_dir = self.workspace.get_temp_dir() / "results" / result_hash.hex()
                tmp_dir.mkdir(parents=True, exist_ok=True)
                key.properties.serializer.save(value, tmp_dir)
                self.result_store[result_hash] = tmp_dir
                with contextlib.suppress(FileNotFoundError):
                    shutil.rmtree(tmp_dir)

    def __delitem__(self, key: Task[R], /) -> None:
        try:
            result_hash = key.result_hash(workspace=self.workspace)
            del self.result_store[result_hash]
        except Exception as e:
            msg = f"Result for task {key} not found in cache."
            raise KeyError(msg) from e

    def __iter__(self) -> Iterator[Task]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.result_store)

    def __contains__(self, key: object, /) -> bool:
        if not isinstance(key, Task):
            return False
        try:
            result_hash = key.result_hash(workspace=self.workspace)
        except RuntimeError:
            return False
        return result_hash in self.result_store
