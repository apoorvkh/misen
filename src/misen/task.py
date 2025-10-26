from __future__ import annotations

import itertools
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    ParamSpec,
    TypeVar,
    cast,
)

import dill
from misen_serialization import canonical_hash
from msgspec import Struct

if TYPE_CHECKING:
    from concurrent.futures import Future
    from pathlib import Path

    from .executor import Executor
    from .workspace import ResultHash, Workspace

__all__ = ["Task", "properties", "resources"]

P = ParamSpec("P")
R = TypeVar("R")


class TaskProperties(Struct, frozen=True):
    id: str
    cache: bool = False
    version: int = 0
    exclude: set[str] = set()
    defaults: dict[str, Any] = {}
    to_bytes: Callable[[Any], bytes] = dill.dumps
    from_bytes: Callable[[bytes], Any] = dill.loads


def properties(
    id: str | None = None,  # TODO: command to fill these in when None
    cache: bool = False,
    version: int = 0,
    exclude: set[str] = set(),
    defaults: dict[str, Any] = {},
    to_bytes: Callable[[R], bytes] = dill.dumps,  # TODO: typing
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
                version=version,
                exclude=exclude,
                defaults=defaults,
                to_bytes=to_bytes,
                from_bytes=from_bytes,
            ),
        )
        return func

    return decorator


class TaskResources(Struct):
    time: int | None = None  # walltime (minutes)
    nodes: int = 1  # number of nodes
    memory: int = 8  # memory in GB
    cpus: int = 1  # number of logical cores
    gpus: int = 0  # number of GPUs
    gpu_memory: int | None = None  # memory (GB) per GPU


def resources(
    time: int | None = None,  # walltime (minutes)
    nodes: int = 1,  # number of nodes
    memory: int = 8,  # memory in GB
    cpus: int = 1,  # number of logical cores
    gpus: int = 0,  # number of GPUs
    gpu_memory: int | None = None,  # memory (GB) per GPU
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        setattr(
            func,
            "__task_resources__",
            TaskResources(
                time=time,
                nodes=nodes,
                memory=memory,
                cpus=cpus,
                gpus=gpus,
                gpu_memory=gpu_memory,
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
        self.resources = getattr(func, "__task_resources__", TaskResources())

        self.func = func
        self.args = args
        self.kwargs = kwargs

    @property
    def T(self) -> R:
        return cast("R", self)

    def is_cached(self, workspace: Workspace | None = None) -> bool:
        if workspace is None:
            from .workspace import Workspace

            workspace = Workspace.auto()

        return self.properties.cache and self in workspace.results

    @property
    def _dependencies(self) -> list[Task]:
        return [t for t in itertools.chain(self.args, self.kwargs.values()) if isinstance(t, Task)]

    def run(self, workspace: Workspace | None = None, executor: Executor | None = None) -> Future:
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

        if self.properties.cache:
            if (serialized_result := workspace.results.get(self)) is not None:
                return serialized_result.value()

            if not compute_if_uncached:
                raise RuntimeError(f"{self} is not cached.")

        if not compute_uncached_deps and any(
            not t.is_cached(workspace=workspace) for t in self._dependencies
        ):
            raise RuntimeError(f"{self} has dependencies which must be computed and cached first.")

        result = self.func(
            *tuple(
                v.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True)
                if isinstance(v, Task)
                else v
                for v in self.args
            ),
            **{
                k: v.result(
                    workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True
                )
                if isinstance(v, Task)
                else v
                for k, v in self.kwargs.items()
            },
        )  # type: ignore

        resolved_hash = self._resolved_hash(workspace=workspace)
        workspace._result_hash_cache[resolved_hash] = cast("ResultHash", canonical_hash(result))

        if self.properties.cache:
            from .workspace import SerializedResult

            workspace.results[self] = SerializedResult(
                deserializer=self.properties.from_bytes,
                data=self.properties.to_bytes(result),
            )

        return result

    def logs(self, workspace: Workspace | None = None) -> str:
        if workspace is None:
            from .workspace import Workspace

            workspace = Workspace.auto()

        return workspace.logs[self]

    def work_dir(self, workspace: Workspace | None = None) -> Path:
        if workspace is None:
            from .workspace import Workspace

            workspace = Workspace.auto()

        return workspace.get_work_dir(task=self)

    def __repr__(self):
        return "".join(
            [
                f"Task({self.func.__module__}.{self.func.__qualname__}",
                *(f", {a.__repr__()}" for a in self.args),
                *(f", {k}={v.__repr__()}" for k, v in self.kwargs.items()),
                f", hash={self.__hash__()}",
                ")",
            ]
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
        if not hasattr(self, "_hash"):
            self._hash = canonical_hash(
                (
                    self.properties.id,
                    self.properties.version,
                    {
                        k: (v.__hash__() if isinstance(v, Task) else canonical_hash(v))
                        for k, v in self._arguments_for_hashing.items()
                    },
                )
            )
        return self._hash

    def _resolved_hash(self, workspace: Workspace) -> ResolvedTaskHash:
        """A hash that represents the Task object using its resolved arguments."""
        task_hash: TaskHash = cast("TaskHash", self.__hash__())

        # fast session-only cache
        if (resolved_hash := workspace._resolved_hashes.get(task_hash)) is not None:
            return resolved_hash

        # slower workspace cache
        if (resolved_hash := workspace._resolved_hash_cache.get(task_hash)) is None:
            hashed_arguments = {
                k: (
                    v._result_hash(workspace=workspace)
                    if isinstance(v, Task)
                    else canonical_hash(v)
                )
                for k, v in self._arguments_for_hashing.items()
            }
            resolved_hash = cast(
                "ResolvedTaskHash",
                canonical_hash((self.properties.id, self.properties.version, hashed_arguments)),
            )
            workspace._resolved_hash_cache[task_hash] = resolved_hash

        workspace._resolved_hashes[task_hash] = resolved_hash
        return resolved_hash

    def _result_hash(self, workspace: Workspace) -> ResultHash:
        """Hash of the task's result object (getter from workspace cache)."""
        """Raises RuntimeError if the task has not been computed."""
        # fast session-only cache
        task_hash: TaskHash = cast("TaskHash", self.__hash__())
        if (result_hash := workspace._result_hashes.get(task_hash)) is not None:
            return result_hash

        # slower workspace cache
        resolved_hash: ResolvedTaskHash = self._resolved_hash(workspace=workspace)
        if (result_hash := workspace._result_hash_cache.get(resolved_hash)) is None:
            raise RuntimeError(f"Task {self} must be computed first.")

        workspace._result_hashes[task_hash] = result_hash
        return result_hash


class TaskHash(int): ...


class ResolvedTaskHash(int): ...


class ResultHash(int): ...


class SerializedResult(Generic[R]):
    def __init__(self, deserializer: Callable[[bytes], R], data: bytes):
        self.deserializer = deserializer
        self.data = data

    def value(self) -> R:
        return self.deserializer(self.data)
