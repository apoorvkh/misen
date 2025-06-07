from __future__ import annotations

import itertools
import sys
from inspect import signature
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Generic, ParamSpec, TypeVar, cast

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
        "properties",
        "func",
        "args",
        "kwargs",
        "arguments_for_hashing",
        "_object_hash",
        "_initialized",
    )

    def __init__(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        self.properties = getattr(
            func,
            "__task_properties__",
            TaskProperties(f"{func.__module__}.{func.__qualname__}"),
        )

        self.func = func
        self.args = args
        self.kwargs = MappingProxyType(kwargs)

        bound_arguments = signature(func).bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        self.arguments_for_hashing = MappingProxyType(
            {
                k: v
                for k, v in bound_arguments.arguments.items()
                if k not in self.properties.exclude
                and (k not in self.properties.defaults or self.properties.defaults[k] != v)
            }
        )

        self._object_hash = self.__object_hash__()

        self._initialized = True

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_initialized", False) and name in self.__slots__:
            raise AttributeError(f"Cannot modify Task.{name} after initialization")
        super().__setattr__(name, value)

    @property
    def T(self) -> R:
        return self  # type: ignore

    def __repr__(self):
        return "".join(
            [
                f"Task({self.func.__module__}.{self.func.__qualname__}",
                *(f", {a.__repr__()}" for a in self.args),
                *(f", {k}={v.__repr__()}" for k, v in self.kwargs.items()),
                ")",
            ]
        )

    def deps_cached(self, workspace: Workspace) -> bool:
        return all(
            workspace.is_cached(task=t)
            for t in itertools.chain(self.args, self.kwargs.values())
            if isinstance(t, Task) and t.properties.cache_result and not t.properties.always_compute
        )

    def result(self, workspace: Workspace | None = None) -> R:
        """Compute or retrieve the Task result and cache it."""
        from .workspace import Workspace  # avoids circular import

        workspace = workspace or Workspace.load()

        if not self.deps_cached(workspace=workspace):
            raise RuntimeError(f"{self} has dependencies which must be computed and cached first.")

        if self.properties.cache_result and not self.properties.always_compute:
            if (result_bytes := workspace.get_result(task=self)) is not None:
                return self.properties.from_bytes(result_bytes)

        result = self.func(
            *tuple(v.result(workspace=workspace) if isinstance(v, Task) else v for v in self.args),
            **{
                k: v.result(workspace=workspace) if isinstance(v, Task) else v
                for k, v in self.kwargs.items()
            },
        )  # type: ignore

        workspace.set_result_hash(task=self, h=cast("ResultHash", _deterministic_hash(result)))
        if self.properties.cache_result:
            workspace.set_result(task=self, result=self.properties.to_bytes(result))

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

    def __hash__(self) -> int:
        """A hash that represents the Task object using its constituent task graph."""
        return self.__object_hash__()

    def __object_hash__(self) -> ObjectHash:
        if hasattr(self, "_object_hash"):
            return self._object_hash
        return cast(
            "ObjectHash",
            _deterministic_hash(
                (
                    self.properties.id,
                    {
                        k: (v.__object_hash__() if isinstance(v, Task) else _deterministic_hash(v))
                        for k, v in self.arguments_for_hashing.items()
                    },
                )
            ),
        )

    def __resolved_hash__(self, workspace: Workspace) -> ResolvedHash:
        """A hash that represents the Task object using its resolved arguments."""
        if (resolved_hash := workspace.get_resolved_hash(task=self)) is None:
            resolved_hash = cast(
                "ResolvedHash",
                _deterministic_hash(
                    (
                        self.properties.id,
                        {
                            k: (
                                _deterministic_hash(v)
                                if not isinstance(v, Task)
                                else int(v.__result_hash__(workspace=workspace))
                            )
                            for k, v in self.arguments_for_hashing.items()
                        },
                    )
                ),
            )
            workspace.set_resolved_hash(task=self, h=resolved_hash)
        return resolved_hash

    def __result_hash__(self, workspace: Workspace) -> ResultHash:
        """Getter for the hash of result, which is computed and stored in result()."""
        if (_result_hash := workspace.get_result_hash(task=self)) is None:
            raise RuntimeError(f"{self} must be computed before retrieving the result hash.")
        return _result_hash


class ObjectHash(int):
    pass


class ResolvedHash(int):
    pass


class ResultHash(int):
    pass


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


def _deterministic_hash(obj: Any, seed: int = 0) -> int:
    serialized_data = msgspec.json.encode(
        (_class_identifier(obj), obj),
        enc_hook=lambda o: (_class_identifier(o), dill.dumps(o)),
        order="sorted",
    )
    return xxh3_64_intdigest(serialized_data, seed=seed)
