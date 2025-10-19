from __future__ import annotations

import inspect
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import MutableMapping
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Mapping, TypeAlias, cast

import dill
from misen_serialization import canonical_hash

from .settings import Settings
from .task import Task

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "Workspace",
    "SerializedResult",
    "ResolvedHashCacheABC",
    "ResultHashCacheABC",
    "ResultCacheABC",
    "LogStoreABC",
]


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


WorkspaceType: TypeAlias = str | Literal["auto", "memory"]


class Workspace(ABC, metaclass=WorkspaceMeta):
    resolved_hashes: ResolvedHashCacheABC
    result_hashes: ResultHashCacheABC
    results: ResultCacheABC
    logs: LogStoreABC

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

        return MemoryWorkspace(i=20)

    @staticmethod
    def _resolve_type(t: WorkspaceType) -> type["Workspace"]:
        match t:
            case "auto":
                return Workspace
            case "memory":
                from misen.workspaces.memory import MemoryWorkspace

                return MemoryWorkspace
            case _:
                module, class_name = t.split(":", maxsplit=1)
                return getattr(import_module(module), class_name)

    def get_work_dir(self, task: Task) -> Path:
        """Return a directory where the task can store working files. E.g. to cache intermediate results."""
        raise NotImplementedError

    def __resolved_hash__(self, task: Task) -> ResolvedHash:
        """A hash that represents the Task object using its resolved arguments."""
        resolved_hash = self.resolved_hashes.get(task)
        if resolved_hash is None:
            resolved_hash = cast(
                "ResolvedHash",
                canonical_hash(
                    (
                        task.properties.id,
                        {
                            k: (
                                self.__result_hash__(task=task)
                                if isinstance(v, Task)
                                else canonical_hash(v)
                            )
                            for k, v in task._arguments_for_hashing.items()
                        },
                    )
                ),
            )
            self.resolved_hashes[task] = resolved_hash
        return resolved_hash

    def __result_hash__(self, task: Task) -> ResultHash:
        """Getter for the hash of result, which is computed and stored in result()."""
        try:
            return self.result_hashes[task]
        except KeyError:
            raise RuntimeError(f"{task} must be computed first.")


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
        try:
            return self.mapping[cast("ObjectHash", key.__hash__())]
        except KeyError as e:
            raise KeyError(f"Task {key} not found in cache") from e

    def __setitem__(self, key: Task, value: ResolvedHash) -> None:
        self.mapping[cast("ObjectHash", key.__hash__())] = value

    def __delitem__(self, key: Task) -> None:
        try:
            del self.mapping[cast("ObjectHash", key.__hash__())]
        except KeyError as e:
            raise KeyError(f"Task {key} not found in cache") from e

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self) -> Iterator[Task]:
        raise NotImplementedError


class ResultHashCacheABC(MutableMapping[Task, ResultHash], ABC):
    mapping: MutableMapping[ResolvedHash, ResultHash]
    workspace: Workspace

    def __getitem__(self, key: Task) -> ResultHash:
        try:
            return self.mapping[self.workspace.__resolved_hash__(task=key)]
        except KeyError as e:
            raise KeyError(f"Task {key} not found in cache") from e

    def __setitem__(self, key: Task, value: ResultHash) -> None:
        self.mapping[self.workspace.__resolved_hash__(task=key)] = value

    def __delitem__(self, key: Task) -> None:
        try:
            del self.mapping[self.workspace.__resolved_hash__(task=key)]
        except KeyError as e:
            raise KeyError(f"Task {key} not found in cache") from e

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self) -> Iterator[Task]:
        raise NotImplementedError


class ResultCacheABC(MutableMapping[Task, SerializedResult], ABC):
    mapping: MutableMapping[ResultHash, bytes]
    workspace: Workspace

    def __getitem__(self, key: Task) -> SerializedResult:
        return dill.loads(self.mapping[self.workspace.__result_hash__(task=key)])

    def __setitem__(self, key: Task, value: SerializedResult) -> None:
        self.mapping[self.workspace.__result_hash__(task=key)] = dill.dumps(value)

    def __delitem__(self, key: Task) -> None:
        del self.mapping[self.workspace.__result_hash__(task=key)]

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self) -> Iterator[Task]:
        raise NotImplementedError

    def __contains__(self, key: object, /) -> bool:
        if not isinstance(key, Task):
            return False
        try:
            result_hash = self.workspace.__result_hash__(task=key)
        except RuntimeError:
            return False
        return result_hash in self.mapping


class LogStoreABC(Mapping[Task, dict[str, str]], ABC):
    workspace: Workspace

    @abstractmethod
    def __getitem__(self, key: Task) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, key: Task, value: dict[str, str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def __delitem__(self, key: Task) -> None:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[Task]:
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, key: object, /) -> bool:
        raise NotImplementedError
