from __future__ import annotations

from abc import ABC
from collections.abc import MutableMapping
from typing import Any, Callable, Iterator, Literal, cast

import dill
import msgspec

from .settings import ConfigABC, ConfigurableABC, Settings
from .task import Task

_WORKSPACES = {}


class WorkspaceConfig(ConfigABC["WorkspaceConfig", "Workspace"], kw_only=True):
    type: str | Literal["memory", "auto"] = "auto"

    @staticmethod
    def settings_key() -> str:
        return "workspace"

    @staticmethod
    def default() -> WorkspaceConfig:
        from .workspaces.memory import MemoryWorkspaceConfig

        return MemoryWorkspaceConfig(i=10)

    def resolve_component_type(self) -> type[Workspace]:
        match self.type:
            case "memory":
                from .workspaces.memory import MemoryWorkspace

                return MemoryWorkspace
        return super().resolve_component_type()

    def load(self, settings: Settings | None = None) -> Workspace:
        if self.__class__ is WorkspaceConfig and self.type != "auto":
            raise TypeError(
                "Cannot load WorkspaceConfig directly unless type='auto'. Use a specific workspace config class."
            )
        return super().load(settings=settings)


class Workspace(ConfigurableABC[WorkspaceConfig]):
    resolved_hashes: ResolvedHashCacheABC
    result_hashes: ResultHashCacheABC
    results: ResultCacheABC

    def __new__(cls, config: WorkspaceConfig):
        _config_dict = msgspec.to_builtins(config) | {
            "type": f"{config.__module__}:{config.__class__.__qualname__}"
        }
        _h = hash(msgspec.json.encode(_config_dict, order="sorted"))
        if _h not in _WORKSPACES:
            _WORKSPACES[_h] = super().__new__(cls)
        return _WORKSPACES[_h]

    def __init__(self, config: WorkspaceConfig):
        if not hasattr(self, "_initialized"):
            super().__init__(config=config)
            self._initialized = True

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
