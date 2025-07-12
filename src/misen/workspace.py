from __future__ import annotations

from abc import ABC
from collections.abc import MutableMapping
from typing import Any, Callable, Iterator, Literal, cast

import dill

from .settings import ComponentABC, ConfigABC
from .task import Task


class WorkspaceConfig(ConfigABC["WorkspaceConfig", "Workspace"], kw_only=True):
    type: str | Literal["memory"] | None = None

    @staticmethod
    def settings_key() -> str:
        return "workspace"

    def default(self) -> WorkspaceConfig:
        from .workspaces.memory import MemoryWorkspaceConfig

        return MemoryWorkspaceConfig(i=10)

    def resolve_component_type(self) -> type[Workspace]:
        match self.type:
            case "memory":
                from .workspaces.memory import MemoryWorkspace

                return MemoryWorkspace
        return super().resolve_component_type()


class Workspace(ComponentABC[WorkspaceConfig]):
    resolved_hashes: ResolvedHashCacheABC
    result_hashes: ResultHashCacheABC
    results: ResultCacheABC

    def is_cached(self, task: Task) -> bool:
        """Check if the result of the task is cached."""
        return task.properties.cache_result and task in self.results

    # def get_logs(self, task):
    #     # TODO: A single task may be run multiple times and therefore have multiple logs.
    #     # How should we store and return logs?
    #     raise NotImplementedError

    # def get_work_dir(self, task):
    #     """Return a directory where the task can store working files. E.g. to cache intermediate results."""
    #     raise NotImplementedError


class ObjectHash(int):
    pass


class ResolvedHash(int):
    pass


class ResultHash(int):
    pass


class SerializedResult:
    def __init__(self, deserializer: Callable[[bytes], Any], data: bytes):
        self.deserializer = deserializer
        self.data = data

    def value(self) -> Any:
        return self.deserializer(self.data)


class ResolvedHashCacheABC(MutableMapping[Task, ResolvedHash], ABC):
    mapping: MutableMapping[ObjectHash, ResolvedHash]
    workspace: Workspace

    def __getitem__(self, key: Task) -> ResolvedHash:
        return self.mapping[cast("ObjectHash", key.__hash__())]

    def __setitem__(self, key: Task, value: ResolvedHash) -> None:
        self.mapping[cast("ObjectHash", key.__hash__())] = value

    def __delitem__(self, key: Task) -> None:
        del self.mapping[cast("ObjectHash", key.__hash__())]

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self) -> Iterator[Task]:
        raise NotImplementedError


class ResultHashCacheABC(MutableMapping[Task, ResultHash], ABC):
    mapping: MutableMapping[ResolvedHash, ResultHash]
    workspace: Workspace

    def __getitem__(self, key: Task) -> ResultHash:
        return self.mapping[key.__resolved_hash__(workspace=self.workspace)]

    def __setitem__(self, key: Task, value: ResultHash) -> None:
        self.mapping[key.__resolved_hash__(workspace=self.workspace)] = value

    def __delitem__(self, key: Task) -> None:
        del self.mapping[key.__resolved_hash__(workspace=self.workspace)]

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self) -> Iterator[Task]:
        raise NotImplementedError


class ResultCacheABC(MutableMapping[Task, SerializedResult], ABC):
    mapping: MutableMapping[ResultHash, bytes]
    workspace: Workspace

    def __getitem__(self, key: Task) -> SerializedResult:
        return dill.loads(self.mapping[key.__result_hash__(workspace=self.workspace)])

    def __setitem__(self, key: Task, value: SerializedResult) -> None:
        self.mapping[key.__result_hash__(workspace=self.workspace)] = dill.dumps(value)

    def __delitem__(self, key: Task) -> None:
        del self.mapping[key.__result_hash__(workspace=self.workspace)]

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self) -> Iterator[Task]:
        raise NotImplementedError
