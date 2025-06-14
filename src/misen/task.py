from __future__ import annotations

import itertools
from inspect import signature
from typing import TYPE_CHECKING, Any, Callable, Generic, ParamSpec, TypeVar, cast

import dill
from msgspec import Struct
from xxhash import xxh3_64_intdigest

from .serialization import serialize

if TYPE_CHECKING:
    from concurrent.futures import Future

    from .caches import ResolvedHash, ResultHash
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
    def __init__(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        self.properties = getattr(
            func,
            "__task_properties__",
            TaskProperties(f"{func.__module__}.{func.__qualname__}"),
        )

        self.func = func
        self.args = args
        self.kwargs = kwargs

    @property
    def T(self) -> R:
        return self  # type: ignore

    def __repr__(self):
        return "".join(
            [
                f"Task({self.func.__module__}.{self.func.__qualname__}",
                *(f", {a.__repr__()}" for a in self.args),
                *(f", {k}={v.__repr__()}" for k, v in self.kwargs.items()),
                # f", hash={self.__hash__()}",
                ")",
            ]
        )

    @property
    def arguments_for_hashing(self) -> dict[str, Any]:
        bound_arguments = signature(self.func).bind(*self.args, **self.kwargs)
        bound_arguments.apply_defaults()
        return {
            k: v
            for k, v in bound_arguments.arguments.items()
            if k not in self.properties.exclude
            and (k not in self.properties.defaults or self.properties.defaults[k] != v)
        }

    def dependencies(self) -> list[Task]:
        return [t for t in itertools.chain(self.args, self.kwargs.values()) if isinstance(t, Task)]

    def deps_cached(self, workspace: Workspace) -> bool:
        return all(
            workspace.is_cached(task=t)
            for t in self.dependencies()
            if t.properties.cache_result and not t.properties.always_compute
        )

    def result(self, workspace: Workspace | None = None) -> R:
        """Compute or retrieve the Task result and cache it."""
        from .caches import SerializedResult

        if workspace is None:
            from .workspace import WorkspaceConfig

            workspace = WorkspaceConfig().load()

        if not self.deps_cached(workspace=workspace):
            raise RuntimeError(f"{self} has dependencies which must be computed and cached first.")

        if self.properties.cache_result and not self.properties.always_compute:
            try:
                if (cached_result := workspace.results.get(self)) is not None:
                    return cached_result.value()
            except (KeyError, RuntimeError):
                pass

        result = self.func(
            *tuple(v.result(workspace=workspace) if isinstance(v, Task) else v for v in self.args),
            **{
                k: v.result(workspace=workspace) if isinstance(v, Task) else v
                for k, v in self.kwargs.items()
            },
        )  # type: ignore

        workspace.result_hashes[self] = cast("ResultHash", _deterministic_hash(result))

        if self.properties.cache_result:
            workspace.results[self] = SerializedResult(
                deserializer=self.properties.from_bytes,
                data=self.properties.to_bytes(result),
            )

        return result

    def run(self, workspace: Workspace | None = None, executor: Executor | None = None) -> Future:
        """
        Submit task to executor to fully execute the task graph.
        """
        from .workspace import WorkspaceConfig  # avoids circular import

        if executor is None:
            from .executor import ExecutorConfig  # avoids circular import

            executor = ExecutorConfig().load()

        if workspace is None:
            from .workspace import WorkspaceConfig

            workspace = WorkspaceConfig().load()

        return executor.submit(task=self, workspace=workspace)

    def __hash__(self) -> int:
        """A hash that represents the Task object using its constituent task graph."""
        if not hasattr(self, "_object_hash"):
            self._object_hash = _deterministic_hash(
                (
                    self.properties.id,
                    {
                        k: (v.__hash__() if isinstance(v, Task) else _deterministic_hash(v))
                        for k, v in self.arguments_for_hashing.items()
                    },
                )
            )
        return self._object_hash

    def __resolved_hash__(self, workspace: Workspace) -> ResolvedHash:
        """A hash that represents the Task object using its resolved arguments."""
        if (resolved_hash := workspace.resolved_hashes.get(self)) is None:
            resolved_hash = cast(
                "ResolvedHash",
                _deterministic_hash(
                    (
                        self.properties.id,
                        {
                            k: (
                                v.__result_hash__(workspace=workspace)
                                if isinstance(v, Task)
                                else _deterministic_hash(v)
                            )
                            for k, v in self.arguments_for_hashing.items()
                        },
                    )
                ),
            )
            workspace.resolved_hashes[self] = resolved_hash
        return resolved_hash

    def __result_hash__(self, workspace: Workspace) -> ResultHash:
        """Getter for the hash of result, which is computed and stored in result()."""
        try:
            return workspace.result_hashes[self]
        except KeyError:
            raise RuntimeError(f"{self} must be computed first.")


def _deterministic_hash(obj: Any, seed: int = 0) -> int:
    return xxh3_64_intdigest(serialize(obj), seed=seed)
