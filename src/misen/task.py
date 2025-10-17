from __future__ import annotations

import itertools
from inspect import signature
from typing import TYPE_CHECKING, Any, Callable, Generic, ParamSpec, TypeVar, cast

import dill
from misen_serialization import canonical_hash
from msgspec import Struct

if TYPE_CHECKING:
    from concurrent.futures import Future

    from .executor import Executor
    from .workspace import ResolvedHash, ResultHash, Workspace

__all__ = ["Task", "task"]

P = ParamSpec("P")
R = TypeVar("R")


class TaskProperties(Struct, frozen=True):
    id: str
    cache: bool = False
    always_compute: bool = False
    version: int = 0
    exclude: set[str] = set()
    defaults: dict[str, Any] = {}
    to_bytes: Callable[[Any], bytes] = dill.dumps
    from_bytes: Callable[[bytes], Any] = dill.loads


def task(
    id: str | None = None,  # openssl rand -base64 3
    cache: bool = False,
    always_compute: bool = False,
    version: int = 0,
    exclude: set[str] = set(),
    defaults: dict[str, Any] = {},
    to_bytes: Callable[[R], bytes] = dill.dumps,
    from_bytes: Callable[[bytes], R] = dill.loads,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    # TODO: handle lambda
    # TODO: Callable has no __qualname__
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        setattr(
            func,
            "__task_properties__",
            TaskProperties(
                id=(id or f"{func.__module__}.{func.__qualname__}"),
                cache=cache,
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
        return cast("R", self)

    def run(
        self,
        workspace: Workspace | None = None,
        executor: Executor | None = None,
    ) -> Future:
        """
        Submit task to executor to fully execute the task graph.
        """

        if executor is None:
            from .executor import Executor

            executor = Executor.auto()

        if workspace is None:
            from .workspace import Workspace

            workspace = Workspace.auto()

        return executor.submit(task=self, workspace=workspace)

    def result(
        self,
        workspace: Workspace | None = None,
        compute_if_uncached: bool = False,
        compute_uncached_deps: bool = False,
    ) -> R:
        """Compute or retrieve the Task result and cache it."""

        if workspace is None:
            from .workspace import Workspace

            workspace = Workspace.auto()

        # TODO: can we move this below the other check?
        if not compute_uncached_deps and not self._dependencies_cached(workspace=workspace):
            raise RuntimeError(f"{self} has dependencies which must be computed and cached first.")

        if self.properties.cache and not self.properties.always_compute:
            try:
                if (cached_result := workspace.results.get(self)) is not None:
                    return cached_result.value()
                else:
                    raise RuntimeError(f"{self} is not cached")
            except (KeyError, RuntimeError) as e:
                if not compute_if_uncached:
                    raise e

        result = self.func(
            *tuple(v.result(workspace=workspace) if isinstance(v, Task) else v for v in self.args),
            **{
                k: v.result(workspace=workspace) if isinstance(v, Task) else v
                for k, v in self.kwargs.items()
            },
        )  # type: ignore

        workspace.result_hashes[self] = cast("ResultHash", canonical_hash(result))

        from .workspace import SerializedResult

        if self.properties.cache:
            workspace.results[self] = SerializedResult(
                deserializer=self.properties.from_bytes,
                data=self.properties.to_bytes(result),
            )

        return result

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
    def _dependencies(self) -> list[Task]:
        return [t for t in itertools.chain(self.args, self.kwargs.values()) if isinstance(t, Task)]

    def _dependencies_cached(self, workspace: Workspace) -> bool:
        return all(
            workspace.is_cached(task=t)
            for t in self._dependencies
            if t.properties.cache and not t.properties.always_compute
        )

    @property
    def _arguments_for_hashing(self) -> dict[str, Any]:
        bound_arguments = signature(self.func).bind(*self.args, **self.kwargs)
        bound_arguments.apply_defaults()
        return {
            k: v
            for k, v in bound_arguments.arguments.items()
            if k not in self.properties.exclude
            and (k not in self.properties.defaults or self.properties.defaults[k] != v)
        }

    def __hash__(self) -> int:
        """A hash that represents the Task object using its constituent task graph."""
        if not hasattr(self, "_object_hash"):
            self._object_hash = canonical_hash(
                (
                    self.properties.id,
                    {
                        k: (v.__hash__() if isinstance(v, Task) else canonical_hash(v))
                        for k, v in self._arguments_for_hashing.items()
                    },
                )
            )
        return self._object_hash

    def __resolved_hash__(self, workspace: Workspace) -> ResolvedHash:
        """A hash that represents the Task object using its resolved arguments."""
        resolved_hash = workspace.resolved_hashes.get(self)
        if resolved_hash is None:
            resolved_hash = cast(
                "ResolvedHash",
                canonical_hash(
                    (
                        self.properties.id,
                        {
                            k: (
                                v.__result_hash__(workspace=workspace)
                                if isinstance(v, Task)
                                else canonical_hash(v)
                            )
                            for k, v in self._arguments_for_hashing.items()
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
