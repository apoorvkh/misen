from __future__ import annotations

import sys
from functools import cache
from inspect import signature
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Generic, ParamSpec, TypeVar

import dill
import msgspec
from msgspec import Struct
from xxhash import xxh3_64_intdigest

if TYPE_CHECKING:
    from concurrent.futures import Future

    from .executor import Executor
    from .workspace import Workspace

__all__ = ["Task", "task"]

P = ParamSpec("P")
R = TypeVar("R")


class TaskProperties(Struct, frozen=True):
    """Dataclass for task properties. Attributes are immutable so that Task.__hash__ will be constant and cacheable."""

    id: str
    cache_result: bool = False
    always_compute: bool = False
    version: int = 0
    exclude: set[str] = set()
    defaults: dict[str, Any] = {}
    to_bytes: Callable[[Any], bytes] = dill.dumps
    from_bytes: Callable[[bytes], Any] = dill.loads


def task(
    id: str | None = None,  # openssl rand -base64 3
    cache_result: bool = False,
    always_compute: bool = False,
    version: int = 0,
    exclude: set[str] = set(),
    defaults: dict[str, Any] = {},
    to_bytes: Callable[[R], bytes] = dill.dumps,
    from_bytes: Callable[[bytes], R] = dill.loads,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    # TODO: handle lambda
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        setattr(
            func,
            "__task_properties__",
            TaskProperties(
                id=(id or f"{func.__module__}.{func.__qualname__}"),
                cache_result=cache_result,
                always_compute=always_compute,
                version=version,
                exclude=exclude,
                defaults=defaults,
                to_bytes=to_bytes,
                from_bytes=from_bytes,
            ),
        )
        return func

    return decorator


class Task(Generic[R]):
    __slots__ = (
        "func",
        "args",
        "kwargs",
        "bound_arguments",
        "properties",
        "_initialized",
    )

    def __init__(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        self.func = func
        self.args = args
        self.kwargs = MappingProxyType(kwargs)

        bound_arguments = signature(func).bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        self.bound_arguments = MappingProxyType(bound_arguments.arguments)

        self.properties = getattr(
            self.func,
            "__task_properties__",
            TaskProperties(f"{func.__module__}.{func.__qualname__}"),
        )

        self._initialized = True

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_initialized", False) and name in self.__slots__:
            raise AttributeError(f"Cannot modify {name} after initialization")
        super().__setattr__(name, value)

    @property
    def T(self) -> R:
        return self  # type: ignore

    def __repr__(self):
        return f"Task(func={self.func.__module__}.{self.func.__qualname__}, arguments={self.bound_arguments})"

    def __hash__(self) -> int:
        return self.__task_hash__()

    @cache
    def __task_hash__(self) -> int:
        """A hash that represents the Task object using its constituent task graph."""
        return _hash(
            (
                self.properties.id,
                {
                    k: (v.__task_hash__() if isinstance(v, Task) else _hash(v))
                    for k, v in self.bound_arguments.items()
                },
            )
        )

    def __resolved_hash__(self, workspace: Workspace) -> int:
        """A hash that represents the Task object using its resolved arguments."""
        _resolved_hash = workspace.resolved_hashes.get(self.__task_hash__(), None)
        if _resolved_hash is None:
            _resolved_hash = _hash(
                (
                    self.properties.id,
                    {
                        k: (
                            _hash(v)
                            if not isinstance(v, Task)
                            else v.__result_hash__(workspace=workspace)
                        )
                        for k, v in self.bound_arguments.items()
                    },
                )
            )
            workspace.resolved_hashes[self.__task_hash__()] = _resolved_hash
        return _resolved_hash

    def __result_hash__(self, workspace: Workspace) -> int:
        """Getter for the hash of result, which is computed and stored in result()."""
        return workspace.result_hashes[self.__resolved_hash__(workspace=workspace)]

    def is_cached(self, workspace: Workspace | None = None) -> bool:
        from .workspace import Workspace

        workspace = workspace or Workspace.load()
        try:
            return (
                self.properties.cache_result
                and self.__result_hash__(workspace=workspace) in workspace.results.keys()
            )
        except (KeyError, RuntimeError):
            return False

    def dependencies(self) -> list[Task]:
        return [arg for arg in self.bound_arguments.values() if isinstance(arg, Task)]

    def deps_cached(self, workspace: Workspace) -> bool:
        return all(
            t.is_cached(workspace=workspace)
            for t in self.dependencies()
            if t.properties.cache_result
        )

    def result(self, workspace: Workspace | None) -> R:
        from .workspace import Workspace  # avoids circular import

        workspace = workspace or Workspace.load()

        if not self.deps_cached(workspace=workspace):
            raise RuntimeError(
                f"Task {self} has dependencies which must be cached before computing the result."
            )

        if self.properties.cache_result and not self.properties.always_compute:
            return self.properties.from_bytes(
                workspace.results[self.__result_hash__(workspace=workspace)]
            )

        result = self.func(
            *tuple(v.result(workspace=workspace) if isinstance(v, Task) else v for v in self.args),
            **{
                k: v.result(workspace=workspace) if isinstance(v, Task) else v
                for k, v in self.kwargs.items()
            },
        )  # type: ignore

        result_hash = _hash(result)
        workspace.result_hashes[self.__resolved_hash__(workspace=workspace)] = result_hash
        if self.properties.cache_result:
            workspace.results[result_hash] = self.properties.to_bytes(result)

        return result

    def run(self, workspace: Workspace | None = None, executor: Executor | None = None) -> Future:
        """
        Submit task to executor to fully execute the task graph.
        """
        from .executor import Executor  # avoids circular import
        from .workspace import Workspace  # avoids circular import

        executor = executor or Executor.load()
        workspace = workspace or Workspace.load()
        return executor.submit(task=self, workspace=workspace)

    # @property
    # def work_dir(self, workspace: Workspace | None = None):
    #     from .workspace import Workspace  # avoids circular import

    #     workspace = workspace or Workspace.load()
    #     return workspace.get_work_dir(self)


def _class_identifier(cls_or_obj: type | Any) -> str:
    cls = cls_or_obj if isinstance(cls_or_obj, type) else type(cls_or_obj)
    class_name = cls.__qualname__
    module_name = cls.__module__
    if module_name == "__main__":
        module = sys.modules["__main__"]
        if hasattr(module, "__file__") and module.__file__ is not None:
            module_name = module.__file__.split("/")[-1].split(".")[0]
        else:
            return class_name
    return f"{module_name}.{cls.__qualname__}"


def _hash(obj: Any, seed: int = 0) -> int:
    serialized_data = msgspec.json.encode(
        (_class_identifier(obj), obj),
        enc_hook=lambda o: (_class_identifier(o), dill.dumps(o)),
        order="sorted",
    )
    return xxh3_64_intdigest(serialized_data, seed=seed)
