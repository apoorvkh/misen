from __future__ import annotations

import inspect
from abc import ABC, ABCMeta
from collections.abc import MutableMapping
from importlib import import_module
from typing import Any, Callable, Iterator, Literal, cast

import dill

from .settings import Settings
from .task import Task, _deterministic_hash


class WorkspaceMeta(ABCMeta):
    """
    Metaclass that turns every subclass into a *parameterised singleton* and
    makes `inspect.signature(SubClass)` show the parameters of `SubClass.__init__`.
    """

    _instances = {}

    def __new__(mcls, name, bases, namespace, **kwds):
        cls = super().__new__(mcls, name, bases, namespace, **kwds)
        init_sig = inspect.signature(cls.__init__)
        params = list(init_sig.parameters.values())[1:]
        cls.__signature__ = init_sig.replace(parameters=params)  # type: ignore
        return cls

    def __call__(cls, **kwargs):
        key = _deterministic_hash(kwargs)

        if key not in WorkspaceMeta._instances:
            WorkspaceMeta._instances[key] = super().__call__(**kwargs)
        return WorkspaceMeta._instances[key]


WorkspaceType = str | Literal["auto", "memory", "disk"]


class Workspace(ABC, metaclass=WorkspaceMeta):
    resolved_hashes: ResolvedHashCacheABC
    result_hashes: ResultHashCacheABC
    results: ResultCacheABC

    @staticmethod
    def resolve_type(t: WorkspaceType) -> type["Workspace"]:
        match t:
            case "auto":
                return Workspace
            case "memory":
                from misen.workspaces.memory import MemoryWorkspace

                return MemoryWorkspace
            case "disk":
                from misen.workspaces.disk import DiskWorkspace

                return DiskWorkspace
            case _:
                module, class_name = t.split(":", maxsplit=1)
                return getattr(import_module(module), class_name)

    @staticmethod
    def auto(settings: Settings | None = None) -> "Workspace":
        if settings is None:
            settings = Settings()

        workspace_type = settings.toml_data.get("workspace_type", "auto")
        workspace_cls = Workspace.resolve_type(workspace_type)
        if workspace_cls is not Workspace:
            return workspace_cls(**settings.toml_data.get("workspace_kwargs", {}))

        # default
        from misen.workspaces.memory import MemoryWorkspace

        return MemoryWorkspace(i=20)

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


class Hash(int):
    pass


class ObjectHash(Hash):
    pass


class ResolvedHash(Hash):
    pass


class ResultHash(Hash):
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
