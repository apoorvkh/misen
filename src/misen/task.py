from __future__ import annotations

import itertools
from contextlib import nullcontext
from functools import cache
from inspect import signature
from time import time_ns
from typing import TYPE_CHECKING, Any, Generic, Literal, ParamSpec, TypeVar, cast

from msgspec import Struct
from typing_extensions import assert_never

from .utils.auto import resolve_auto
from .utils.graph import DependencyGraph
from .utils.hashes import ResolvedTaskHash, ResultHash, TaskHash, short_hash
from .utils.log_capture import capture_all_output
from .utils.object_io import DefaultSerializer, Serializer
from .utils.sentinels import WORK_DIR

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from .executor import Executor
    from .workspace import Workspace

__all__ = ["Task", "resources", "task"]

P = ParamSpec("P")
R = TypeVar("R")


# TODO: walk dependency structures for Tasks
# TODO: signature(self.func).bind(...) is slow?


class Task(Generic[R]):
    def __init__(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> None:
        self.properties: TaskProperties = getattr(
            func,
            "__task_properties__",
            TaskProperties(f"{func.__module__}.{func.__qualname__}"),  # ty:ignore[unresolved-attribute]
        )
        self.resources: TaskResources = getattr(func, "__task_resources__", TaskResources())

        self.func: Callable[P, R] = func
        self.args: P.args = args
        self.kwargs: P.kwargs = kwargs

    def __repr__(self) -> str:
        return (
            f"Task({self.func.__module__}.{self.func.__qualname__}, "
            f"hash={short_hash(self)}){' [C]' if self.properties.cache else ''}"
        )

    @property
    def T(self) -> R:  # noqa: N802
        return cast("R", self)

    @property
    def dependencies(self) -> set[Task]:
        return {t for t in itertools.chain(self.args, self.kwargs.values()) if isinstance(t, Task)}

    def is_cached(self, workspace: Workspace | Literal["auto"] = "auto") -> bool:
        workspace = resolve_auto(workspace=workspace)
        return self.properties.cache and self in workspace.results

    def uncached_deps(self, workspace: Workspace | Literal["auto"] = "auto") -> Iterator[Task]:
        return filter(lambda t: t.properties.cache and not t.is_cached(workspace=workspace), self.dependencies)

    def are_deps_cached(self, workspace: Workspace | Literal["auto"] = "auto") -> bool:
        return all(self.uncached_deps(workspace=workspace))

    def done(self, workspace: Workspace) -> bool:
        try:
            workspace.get_result_hash(task=self)
        except RuntimeError:
            return False
        return True

    def is_running(self, workspace: Workspace) -> bool:
        if not self.are_deps_cached(workspace=workspace):
            return False
        return workspace.lock(namespace="task", key=self.resolved_hash(workspace=workspace).hex()).is_locked()

    def status(self, workspace: Workspace | Literal["auto"] = "auto") -> Literal["running", "done", "unknown"]:
        workspace = resolve_auto(workspace=workspace)

        if self.done(workspace=workspace):
            return "done"
        if self.is_running(workspace=workspace):
            return "running"
        return "unknown"

    def dependency_graph(
        self,
        *,
        exclude_cacheable: bool = False,
        exclude_cached: bool = False,
        workspace: Workspace | Literal["auto"] = "auto",
    ) -> DependencyGraph[Task]:
        if exclude_cached:
            workspace = resolve_auto(workspace=workspace)

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

            for dep in task.dependencies:
                if not _include(dep):
                    continue
                graph.add_edge(task_node, _get_node(dep), None)
                if dep not in seen:
                    seen.add(dep)
                    stack.append(dep)

        return graph

    def run(self, *, workspace: Workspace | Literal["auto"] = "auto", executor: Executor | Literal["auto"] = "auto"):
        """
        Submit task to an executor and return a mapping from each distributable task to its
        dependency set and runtime job status handle.
        """
        executor = resolve_auto(executor=executor)
        workspace = resolve_auto(workspace=workspace)
        return executor.submit(tasks={self}, workspace=workspace)

    def result(
        self,
        *,
        workspace: Workspace | Literal["auto"] = "auto",
        compute_if_uncached: bool = False,
        compute_uncached_deps: bool = False,
    ) -> R:
        """Compute or retrieve the Task result and cache it."""
        workspace = resolve_auto(workspace=workspace)

        if self.properties.cache:
            if (result := workspace.results.get(self)) is not None:
                return result

            if not compute_if_uncached:
                msg = f"{self} is not cached."
                raise RuntimeError(msg)

        if not compute_uncached_deps and not self.are_deps_cached(workspace=workspace):
            uncached_deps = list(self.uncached_deps(workspace=workspace))
            msg = f"{self} has dependencies which must be computed and cached first: {uncached_deps}"
            raise RuntimeError(msg)

        dep_results: dict[Task, Any] = {
            t: t.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True)
            for t in self.dependencies
        }

        def resolve_argument(dep: Task | Any) -> Any:
            if isinstance(dep, Task):
                return dep_results[dep]
            if dep is WORK_DIR:
                return workspace.get_work_dir(task=self)
            return dep

        resolved_hash = self.resolved_hash(workspace=workspace)

        with (
            workspace.lock(namespace="task", key=resolved_hash.hex()).context(blocking=False)
            if self.properties.cache
            else nullcontext()
        ):
            with capture_all_output(workspace.open_log(task=self, mode="a", timestamp=time_ns())):
                args = (resolve_argument(v) for v in self.args)
                kwargs = {k: resolve_argument(v) for k, v in self.kwargs.items()}
                result = self.func(*args, **kwargs)

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

    def work_dir(self, workspace: Workspace | Literal["auto"] = "auto") -> Path:
        workspace = resolve_auto(workspace=workspace)
        return workspace.get_work_dir(task=self)

    def __hash__(self) -> int:
        return hash(int(self.task_hash()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Task):
            return NotImplemented
        return self.task_hash() == other.task_hash()

    def task_hash(self) -> TaskHash:
        """A hash that represents the Task object using its constituent task graph."""
        if not hasattr(self, "_cached_task_hash"):
            self._cached_task_hash = TaskHash.from_object(
                (
                    self.properties.id,
                    self.properties.version,
                    self._hashed_arguments(),
                ),
            )
        return self._cached_task_hash

    def resolved_hash(self, workspace: Workspace) -> ResolvedTaskHash:
        """A hash that represents the Task object using its resolved arguments."""
        resolved_hash: ResolvedTaskHash | None = workspace.get_resolved_hash(self)

        if resolved_hash is None:
            resolved_hash = ResolvedTaskHash.from_object(
                (
                    self.properties.id,
                    self.properties.version,
                    self._hashed_arguments(hash_task_by_result=True, workspace=workspace),
                ),
            )

            workspace.set_resolved_hash(self, resolved_hash)

        return resolved_hash

    def result_hash(self, workspace: Workspace) -> ResultHash:
        """Hash of the task's result object (getter from workspace cache)."""
        """Raises RuntimeError if the task has not been computed."""
        return workspace.get_result_hash(self)

    def _hashed_arguments(
        self,
        *,
        hash_task_by_result: bool = False,
        workspace: Workspace | Literal["auto"] = "auto",
    ) -> dict[str, tuple[TaskHash | ResultHash, int]]:
        bound_arguments = signature(self.func).bind(*self.args, **self.kwargs)
        bound_arguments.apply_defaults()

        def resolve(key: str, value: Any) -> tuple[TaskHash | ResultHash, int]:
            h = (
                (value.result_hash(workspace=workspace) if hash_task_by_result else value.task_hash())
                if isinstance(value, Task)
                else ResultHash.from_object(value)
            )
            version = self.properties.versions.get((key, h), 0)
            return h, version

        return {
            k: resolve(k, v)
            for k, v in bound_arguments.arguments.items()
            if k not in self.properties.exclude
            and (k not in self.properties.defaults or self.properties.defaults[k] != v)
        }


def task(
    *,
    id: str | None,  # TODO: command to fill these in when None
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
        func.__task_properties__ = TaskProperties(  # ty:ignore[unresolved-attribute]
            id=(id or f"{func.__module__}.{func.__qualname__}"),  # ty:ignore[unresolved-attribute]
            cache=cache,
            version=version,
            exclude=(exclude or set()),
            defaults=(defaults or {}),
            versions={
                (name, ResultHash.from_object(value)): vs
                for name, vv in (versions or {}).items()
                for value, vs in vv.items()
            },
            index_by=index_by,
            serializer=serializer,
        )
        return func

    return decorator


# TODO: resources as a function of arguments


def resources(
    *,
    time: int | None = None,  # walltime (minutes)
    nodes: int = 1,  # number of nodes
    memory: int = 8,  # memory in GB
    cpus: int = 1,  # number of logical cores
    gpus: int = 0,  # number of GPUs
    gpu_memory: int | None = None,  # memory (GB) per GPU
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        func.__task_resources__ = TaskResources(  # ty:ignore[unresolved-attribute]
            time=time,
            nodes=nodes,
            memory=memory,
            cpus=cpus,
            gpus=gpus,
            gpu_memory=gpu_memory,
        )
        return func

    return decorator


class TaskProperties(Struct, frozen=True):
    id: str
    cache: bool = False
    version: int = 0
    exclude: set[str] = set()
    defaults: dict[str, Any] = {}
    versions: dict[tuple[str, ResultHash], int] = {}
    index_by: Literal["task", "result"] = "result"
    serializer: type[Serializer] = DefaultSerializer


class TaskResources(Struct):
    time: int | None = None  # walltime (minutes)
    nodes: int = 1  # number of nodes
    memory: int = 8  # memory in GB
    cpus: int = 1  # number of logical cores
    gpus: int = 0  # number of GPUs
    gpu_memory: int | None = None  # memory (GB) per GPU
