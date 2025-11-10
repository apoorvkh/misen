from __future__ import annotations

import inspect
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import MutableMapping
from importlib import import_module
from typing import (
    TYPE_CHECKING,
    Iterator,
    Literal,
    TypeAlias,
    TypeVar,
    cast,
)

import dill
from misen_serialization import canonical_hash

from .settings import Settings
from .task import ResolvedTaskHash, ResultHash, SerializedResult, Task, TaskHash

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["Workspace"]

R = TypeVar("R")


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
        key = canonical_hash(kwargs)

        if key not in WorkspaceMeta._instances:
            WorkspaceMeta._instances[key] = super().__call__(**kwargs)
        return WorkspaceMeta._instances[key]


WorkspaceType: TypeAlias = str | Literal["auto", "memory", "disk"]


# TODO: figure out better way to avoid serializing workspaces?


class WorkspaceParameters:
    def __init__(self, type: type[Workspace], *args, **kwargs):
        self.type = type
        self.args = args
        self.kwargs = kwargs

    def construct(self) -> Workspace:
        return self.type(*self.args, **self.kwargs)


# TODO: support polling task status from Workspace
# TODO: add task lock to Workspace


class Workspace(ABC, metaclass=WorkspaceMeta):
    def __init__(
        self,
        resolved_hash_cache: MutableMapping[TaskHash, ResolvedTaskHash],
        result_hash_cache: MutableMapping[ResolvedTaskHash, ResultHash],
        result_cache: MutableMapping[ResultHash, bytes],
    ):
        # session-only (non-persistent) caches
        self._resolved_hashes: dict[TaskHash, ResolvedTaskHash] = {}
        self._result_hashes: dict[TaskHash, ResultHash] = {}

        # workspace caches
        self._resolved_hash_cache = resolved_hash_cache
        self._result_hash_cache = result_hash_cache
        self._result_cache = result_cache

        # public accessors to workspace data
        self.results: MutableMapping[Task, SerializedResult] = ResultMap(workspace=self)

    @staticmethod
    def auto(settings: Settings | None = None) -> "Workspace":
        if settings is None:
            settings = Settings()

        workspace_type = settings.toml_data.get("workspace_type", "auto")
        workspace_cls = Workspace._resolve_type(workspace_type)
        if workspace_cls is not Workspace:
            return workspace_cls(**settings.toml_data.get("workspace_kwargs", {}))

        # default
        from misen.workspaces.memory import MemoryWorkspace

        return MemoryWorkspace()

    @abstractmethod
    def to_params(self) -> WorkspaceParameters: ...

    @staticmethod
    def _resolve_type(t: WorkspaceType) -> type["Workspace"]:
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

    @abstractmethod
    def get_work_dir(self, task: Task) -> Path:
        """Return a directory where the task can store working files. E.g. to cache intermediate results."""
        ...


class ResultMap(MutableMapping[Task, SerializedResult]):
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def __getitem__(self, key: Task[R], /) -> SerializedResult[R]:
        try:
            result_hash = key._result_hash(workspace=self.workspace)
            result = self.workspace._result_cache[result_hash]
        except Exception as e:
            raise KeyError(f"Result for task {key} not found in cache.") from e
        return cast("SerializedResult", dill.loads(result))

    def __setitem__(self, key: Task[R], value: SerializedResult[R], /) -> None:
        result_hash = key._result_hash(workspace=self.workspace)
        self.workspace._result_cache[result_hash] = dill.dumps(value)

    def __delitem__(self, key: Task, /) -> None:
        result_hash = key._result_hash(workspace=self.workspace)
        del self.workspace._result_cache[result_hash]

    def __iter__(self) -> Iterator[Task]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.workspace._result_cache)

    def __contains__(self, key: object, /) -> bool:
        if not isinstance(key, Task):
            return False
        try:
            result_hash = key._result_hash(workspace=self.workspace)
        except RuntimeError:
            return False
        return result_hash in self.workspace._result_cache


# TODO: decide how to store logs
