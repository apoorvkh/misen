from __future__ import annotations

import itertools
from contextlib import nullcontext
from functools import cache
from inspect import signature
from time import time_ns
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, ParamSpec, TypeVar, cast

from msgspec import Struct
from typing_extensions import assert_never

from .utils.graph import DependencyGraph
from .utils.hashes import ResolvedTaskHash, ResultHash, TaskHash, short_hash
from .utils.log_capture import capture_all_output
from .utils.object_io import DefaultSerializer, Serializer

if TYPE_CHECKING:
    from pathlib import Path

    from .executor import Executor
    from .workspace import Workspace

__all__ = ["Task", "task", "resources"]

P = ParamSpec("P")
R = TypeVar("R")


class TaskProperties(Struct, frozen=True):
    id: str
    cache: bool = False
    version: int = 0
    exclude: set[str] = set()
    defaults: dict[str, Any] = {}
    versions: dict[str, dict[Any, int]] = {}
    index_by: Literal["task", "result"] = "result"
    serializer: type[Serializer] = DefaultSerializer

    def get_argument_version(self, key: str, value: Any) -> int:
        try:
            return self.versions[key][value]
        except (KeyError, TypeError):
            return 0


def task(
    id: str | None = None,  # TODO: command to fill these in when None
    cache: bool = False,
    version: int = 0,
    exclude: set[str] | None = None,
    defaults: dict[str, Any] | None = None,
    versions: dict[str, dict[Any, int]] | None = None,
    index_by: Literal["task", "result"] = "result",
    serializer: type[Serializer[R]] = DefaultSerializer,  # TODO: typing
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
                exclude=(exclude or set()),
                defaults=(defaults or {}),
                versions=(versions or {}),
                index_by=index_by,
                serializer=serializer,
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
            TaskResources(time=time, nodes=nodes, memory=memory, cpus=cpus, gpus=gpus, gpu_memory=gpu_memory),
        )
        return func

    return decorator


# TODO: walk dependency structures for Tasks
# TODO: signature(self.func).bind(...) is slow?


class Task(Generic[R]):
    def __init__(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        self.properties: TaskProperties = getattr(
            func,
            "__task_properties__",
            TaskProperties(f"{func.__module__}.{func.__qualname__}"),  # ty:ignore[unresolved-attribute]
        )
        self.resources: TaskResources = getattr(func, "__task_resources__", TaskResources())

        self.func: Callable[P, R] = func
        self.args: P.args = args
        self.kwargs: P.kwargs = kwargs

    def __repr__(self):
        return f"Task({self.func.__module__}.{self.func.__qualname__}, hash={short_hash(self)}){' [C]' if self.properties.cache else ''}"

    @property
    def T(self) -> R:
        return cast("R", self)

    def is_cached(self, workspace: Workspace | None = None) -> bool:
        if workspace is None:
            from .workspace import Workspace

            workspace = Workspace.auto()

        return self.properties.cache and self in workspace.results

    def status(self, workspace: Workspace | None = None) -> Literal["running", "done", "unknown"]:
        if workspace is None:
            from .workspace import Workspace

            workspace = Workspace.auto()

        if workspace.lock(namespace="task", key=self._resolved_hash(workspace=workspace).hex()).is_locked():
            return "running"

        if (
            self.properties.cache is False and self._resolved_hash(workspace=workspace) in workspace._result_hash_cache
        ) or (self.properties.cache and self.is_cached(workspace=workspace)):
            return "done"

        return "unknown"

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

        return executor.submit(tasks={self}, workspace=workspace)

    def result(
        self, workspace: Workspace | None = None, compute_if_uncached: bool = False, compute_uncached_deps: bool = False
    ) -> R:
        """Compute or retrieve the Task result and cache it."""

        if workspace is None:
            from .workspace import Workspace

            workspace = Workspace.auto()

        if self.properties.cache:
            if (result := workspace.results.get(self)) is not None:
                return result

            if not compute_if_uncached:
                raise RuntimeError(f"{self} is not cached.")

        if not compute_uncached_deps:
            uncached_deps = [
                t for t in self._dependencies if t.properties.cache and not t.is_cached(workspace=workspace)
            ]
            if len(uncached_deps) > 0:
                raise RuntimeError(f"{self} has dependencies which must be computed and cached first: {uncached_deps}")

        dep_results: dict[Task, Any] = {
            t: t.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True)
            for t in self._dependencies
        }

        def get_value(dep: Task | Any) -> Any:
            return dep_results[dep] if isinstance(dep, Task) else dep

        resolved_hash = self._resolved_hash(workspace=workspace)

        with (
            workspace.lock(namespace="task", key=resolved_hash.hex()).context(blocking=False)
            if self.properties.cache
            else nullcontext()
        ):
            with capture_all_output(workspace.open_log(task=self, mode="a", timestamp=time_ns())):
                result = self.func(
                    *(get_value(v) for v in self.args),
                    **{k: get_value(v) for k, v in self.kwargs.items()},
                )

            match self.properties.index_by:
                case "task":
                    index = resolved_hash
                case "result":
                    index = result
                case _:
                    assert_never(self.properties.index_by)

            workspace.set_result_hash(self, ResultHash.from_object(index))

            if self.properties.cache:
                workspace.results[self] = result

        return result

    def work_dir(self, workspace: Workspace | None = None) -> Path:
        if workspace is None:
            from .workspace import Workspace

            workspace = Workspace.auto()

        return workspace.get_work_dir(task=self)

    @property
    def _arguments_for_hashing(self) -> dict[str, Task | tuple[Any, int]]:
        bound_arguments = signature(self.func).bind(*self.args, **self.kwargs)
        bound_arguments.apply_defaults()
        return {
            k: v if isinstance(v, Task) else (v, self.properties.get_argument_version(k, v))
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
            self._cached_task_hash = TaskHash.from_object(
                (
                    self.properties.id,
                    self.properties.version,
                    {
                        k: (v._task_hash() if isinstance(v, Task) else ResultHash.from_object(v))
                        for k, v in self._arguments_for_hashing.items()
                    },
                )
            )
        return self._cached_task_hash

    def _resolved_hash(self, workspace: Workspace) -> ResolvedTaskHash:
        """A hash that represents the Task object using its resolved arguments."""
        resolved_hash: ResolvedTaskHash | None = workspace.get_resolved_hash(self)

        if resolved_hash is None:
            resolved_hash = ResolvedTaskHash.from_object(
                (
                    self.properties.id,
                    self.properties.version,
                    {
                        k: (v._result_hash(workspace=workspace) if isinstance(v, Task) else ResultHash.from_object(v))
                        for k, v in self._arguments_for_hashing.items()
                    },
                )
            )

            workspace.set_resolved_hash(self, resolved_hash)

        return resolved_hash

    def _result_hash(self, workspace: Workspace) -> ResultHash:
        """Hash of the task's result object (getter from workspace cache)."""
        """Raises RuntimeError if the task has not been computed."""
        return workspace.get_result_hash(self)
