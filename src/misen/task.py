from __future__ import annotations

import itertools
from functools import cache
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
from msgspec import Struct

from .utils.graph import DependencyGraph
from .utils.hashes import ResolvedTaskHash, ResultHash, TaskHash

if TYPE_CHECKING:
    from pathlib import Path

    from .executor import Executor
    from .workspace import Workspace, WorkspaceParameters

__all__ = ["Task", "task", "resources"]

P = ParamSpec("P")
R = TypeVar("R")

# TODO: consider serializable


class TaskProperties(Struct, frozen=True):
    id: str
    cache: bool = False
    version: int = 0
    exclude: set[str] = set()
    defaults: dict[str, Any] = {}
    serializable: bool = True
    to_bytes: Callable[[Any], bytes] = dill.dumps
    from_bytes: Callable[[bytes], Any] = dill.loads

    def __post_init__(self):
        if self.cache:
            assert self.serializable


# TODO: maybe to/from bytes should default to canonical serialization?


def task(
    id: str | None = None,  # TODO: command to fill these in when None
    cache: bool = False,
    version: int = 0,
    exclude: set[str] = set(),
    defaults: dict[str, Any] = {},
    serializable: bool = True,
    to_bytes: Callable[[R], bytes] = dill.dumps,  # TODO: typing
    from_bytes: Callable[[bytes], R] = dill.loads,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    # TODO: handle lambda
    # TODO: Callable has no __qualname__
    # TODO: handle func.__module__ == "__main__"
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        setattr(
            func,
            "__task_properties__",
            TaskProperties(
                id=(id or f"{func.__module__}.{func.__qualname__}"),  # ty:ignore[unresolved-attribute]
                cache=cache,
                version=version,
                exclude=exclude,
                defaults=defaults,
                serializable=serializable,
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


# TODO: resources as a function of arguments


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


# TODO: walk dependency structures for Tasks
# TODO: signature(self.func).bind(...) is slow?


class Task(Generic[R]):
    def __init__(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        self.properties = getattr(
            func,
            "__task_properties__",
            TaskProperties(f"{func.__module__}.{func.__qualname__}"),  # ty:ignore[unresolved-attribute]
        )
        self.resources = getattr(
            func,
            "__task_resources__",
            TaskResources(),
        )

        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return f"Task({self.func.__module__}.{self.func.__qualname__}, hash={self.__hash__() % 101}){' [C]' if self.properties.cache else ''}"  # ty:ignore[possibly-missing-attribute]

    @property
    def T(self) -> R:
        return cast("R", self)

    def is_cached(self, workspace: Workspace | None = None) -> bool:
        if workspace is None:
            from .workspace import Workspace

            workspace = Workspace.auto()

        return self.properties.cache and self in workspace.results

    @property
    def _dependencies(self) -> set[Task]:
        return {t for t in itertools.chain(self.args, self.kwargs.values()) if isinstance(t, Task)}

    def _dependency_graph(
        self,
        exclude_cacheable: bool = False,
        exclude_cached: bool = False,
        workspace: Workspace | None = None,
    ) -> DependencyGraph[Task]:
        if exclude_cached and workspace is None:
            from .workspace import Workspace

            workspace = Workspace.auto()

        @cache
        def _include(dependency: Task) -> bool:
            if exclude_cacheable:
                return dependency.properties.cache is False
            if exclude_cached:
                return not dependency.is_cached(workspace=workspace)
            return True

        graph: DependencyGraph[Task] = DependencyGraph()
        nodes: dict[Task, int] = {}

        def _get_node(t: Task) -> int:
            i = nodes.get(t)
            if i is None:
                i = nodes[t] = graph.add_node(t)
            return i

        # DFS to traverse Task graph and build DAG

        stack: list[Task] = [self]
        seen: set[Task] = {self}

        while stack:
            task: Task = stack.pop()
            task_node = _get_node(task)

            for dep in task._dependencies:
                if not _include(dep):
                    continue
                graph.add_edge(task_node, _get_node(dep), None)
                if dep not in seen:
                    seen.add(dep)
                    stack.append(dep)

        return graph

    def run(self, workspace: Workspace | None = None, executor: Executor | None = None):
        """
        Submit task to an executor and return a mapping from each distributable task to its
        dependency set and runtime job status handle.
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
        workspace: Workspace | WorkspaceParameters | None = None,
        compute_if_uncached: bool = False,
        compute_uncached_deps: bool = False,
    ) -> R:
        """Compute or retrieve the Task result and cache it."""

        if workspace is None:
            from .workspace import Workspace

            workspace = Workspace.auto()
        else:
            from .workspace import WorkspaceParameters

            if isinstance(workspace, WorkspaceParameters):
                workspace = workspace.construct()

        if self.properties.cache:
            if (serialized_result := workspace.results.get(self)) is not None:
                return serialized_result.value()

            if not compute_if_uncached:
                raise RuntimeError(f"{self} is not cached.")

        if not compute_uncached_deps:
            uncached_deps = [
                t for t in self._dependencies if t.properties.cache and not t.is_cached(workspace=workspace)
            ]
            if len(uncached_deps) > 0:
                raise RuntimeError(f"{self} has dependencies which must be computed and cached first: {uncached_deps}")

        result = cast("Callable[..., R]", self.func)(
            *tuple(
                v.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True)
                if isinstance(v, Task)
                else v
                for v in self.args
            ),
            **{
                k: v.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True)
                if isinstance(v, Task)
                else v
                for k, v in self.kwargs.items()
            },
        )

        resolved_hash = self._resolved_hash(workspace=workspace)
        workspace._result_hash_cache[resolved_hash] = ResultHash.from_object(result)

        if self.properties.cache:
            workspace.results[self] = SerializedResult(
                deserializer=self.properties.from_bytes,
                data=self.properties.to_bytes(result),
            )

        return result

    def work_dir(self, workspace: Workspace | None = None) -> Path:
        if workspace is None:
            from .workspace import Workspace

            workspace = Workspace.auto()

        return workspace.get_work_dir(task=self)

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
        return hash(int(self._task_hash()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Task):
            return NotImplemented
        return self._task_hash() == other._task_hash()

    def _task_hash(self) -> TaskHash:
        """A hash that represents the Task object using its constituent task graph."""
        if not hasattr(self, "_cached_task_hash"):
            hashed_arguments = {
                k: (v._task_hash() if isinstance(v, Task) else TaskHash.from_object(v))
                for k, v in self._arguments_for_hashing.items()
            }
            self._cached_task_hash = TaskHash.from_object(
                (self.properties.id, self.properties.version, hashed_arguments)
            )
        return self._cached_task_hash

    def _resolved_hash(self, workspace: Workspace) -> ResolvedTaskHash:
        """A hash that represents the Task object using its resolved arguments."""
        task_hash = self._task_hash()

        # fast session-only cache
        if (resolved_hash := workspace._resolved_hashes.get(task_hash)) is not None:
            return resolved_hash

        # slower workspace cache
        if (resolved_hash := workspace._resolved_hash_cache.get(task_hash)) is None:
            hashed_arguments = {
                k: (v._result_hash(workspace=workspace) if isinstance(v, Task) else ResultHash.from_object(v))
                for k, v in self._arguments_for_hashing.items()
            }
            resolved_hash = ResolvedTaskHash.from_object(
                (self.properties.id, self.properties.version, hashed_arguments)
            )

            workspace._resolved_hash_cache[task_hash] = resolved_hash

        workspace._resolved_hashes[task_hash] = resolved_hash
        return resolved_hash

    def _result_hash(self, workspace: Workspace) -> ResultHash:
        """Hash of the task's result object (getter from workspace cache)."""
        """Raises RuntimeError if the task has not been computed."""
        # fast session-only cache
        task_hash = self._task_hash()
        if (result_hash := workspace._result_hashes.get(task_hash)) is not None:
            return result_hash

        # slower workspace cache
        resolved_hash: ResolvedTaskHash = self._resolved_hash(workspace=workspace)
        if (result_hash := workspace._result_hash_cache.get(resolved_hash)) is None:
            raise RuntimeError(f"Task {self} must be computed first.")

        workspace._result_hashes[task_hash] = result_hash
        return result_hash


class SerializedResult(Generic[R]):
    def __init__(self, deserializer: Callable[[bytes], R], data: bytes):
        self.deserializer = deserializer
        self.data = data

    def value(self) -> R:
        return self.deserializer(self.data)
