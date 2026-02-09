"""This module provides utilities to create a Task from a Python function and arguments.

1. `Task` is used to wrap a Python function and its arguments.
2. `@task` controls how Tasks are identified and how results are stored (corresponding to the decorated function).
3. `@resources` specifies the hardware resources needed to compute a function.

Tasks may form a dependency graph (a DAG; see `Task.dependency_graph()`), containing edges from Tasks to dependencies.

Tasks can be submitted (`Task.submit()`) to the Executor for scheduled execution of the dependency graph.

`Task.result()` will run necessary and specified computations and return the result, retrieving and writing artifacts
(runtime logs, result, etc.) to the Workspace as needed. We typically prefer to `submit()` before calling `result()`.

Tasks can be identified by: `task_hash()` (by dependency graph) or `resolved_hash()` (by resolved arguments).
Results are identified by `result_hash()`. Any task which computes the same result should have the same `result_hash()`.
"""

from __future__ import annotations

import itertools
from contextlib import nullcontext
from functools import cache
from inspect import Signature, signature
from typing import TYPE_CHECKING, Any, Generic, Literal, ParamSpec, TypeVar, cast

from msgspec import Struct
from typing_extensions import assert_never

from misen.utils.auto import resolve_auto
from misen.utils.graph import DependencyGraph
from misen.utils.hashes import ResolvedTaskHash, ResultHash, TaskHash, short_hash
from misen.utils.log_capture import capture_all_output
from misen.utils.object_io import DefaultSerializer, Serializer
from misen.utils.sentinels import WORK_DIR

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from misen.executor import Executor, Job
    from misen.utils.locks import LockLike
    from misen.workspace import Workspace

__all__ = ["Task", "resources", "task"]

P = ParamSpec("P")
R = TypeVar("R")


class Task(Generic[R]):
    """A Task is a lazy wrapper for a function and its arguments.

    Attributes:
        func: The underlying callable.
        args / kwargs: Arguments to pass to `func`. Any value that is itself a `Task` is considered a dependency.
        properties: TaskProperties metadata (typically attached to `func` via `@task(...)`).
        resources: TaskResources metadata (typically attached to `func` via `@resources(...)`).
    """

    __slots__ = ("_cached_signature", "_cached_task_hash", "args", "func", "kwargs", "properties", "resources")

    def __init__(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> None:
        """Initialize a Task with a function and arguments."""
        # Metadata is attached by decorators; fall back to defaults if absent.
        self.properties: TaskProperties = getattr(
            func,
            "__task_properties__",
            TaskProperties(f"{func.__module__}.{func.__qualname__}"),  # ty:ignore[unresolved-attribute]
        )  # TODO: handling of builtins, imports from other packages
        self.resources: TaskResources = getattr(func, "__task_resources__", TaskResources())

        self.func: Callable[P, R] = func
        self.args: P.args = args
        self.kwargs: P.kwargs = kwargs
        if not self.properties.cache and any(v is WORK_DIR for v in itertools.chain(self.args, self.kwargs.values())):
            msg = "WORK_DIR sentinel can only be used when Task.properties.cache == True."
            raise ValueError(msg)

    def __repr__(self) -> str:
        """Return a short debug representation for the task."""
        return (
            f"Task({self.func.__module__}.{self.func.__qualname__}, "
            f"hash={short_hash(self)}){' [C]' if self.properties.cache else ''}"
        )

    @property
    def T(self) -> R:  # noqa: N802
        """Cast Task to its result type. Useful for type checking compliance when passing Tasks as dependencies."""
        return cast("R", self)

    @property
    def dependencies(self) -> set[Task]:
        """Immediate Task dependencies discovered from args/kwargs."""
        # TODO: nested structures?
        return {t for t in itertools.chain(self.args, self.kwargs.values()) if isinstance(t, Task)}

    def is_cached(self, workspace: Workspace | Literal["auto"] = "auto") -> bool:
        """Check if Task's result is cached in the Workspace."""
        workspace = resolve_auto(workspace=workspace)
        return self.properties.cache and self in workspace.results

    def uncached_deps(self, workspace: Workspace | Literal["auto"] = "auto") -> Iterator[Task]:
        """Yields immediate dependencies that are cacheable but not cached."""
        return filter(lambda t: t.properties.cache and not t.is_cached(workspace=workspace), self.dependencies)

    def are_deps_cached(self, workspace: Workspace | Literal["auto"] = "auto") -> bool:
        """Are all immediate dependencies cached?"""
        return not any(self.uncached_deps(workspace=workspace))  # "if there are not any uncached deps"

    def dependency_graph(
        self,
        *,
        exclude_cacheable: bool = False,
        exclude_cached: bool = False,
        workspace: Workspace | Literal["auto"] = "auto",
    ) -> DependencyGraph[Task]:
        """Build a DependencyGraph rooted at this Task. Directed edges: task -> dependency.

        Args:
            exclude_cacheable: Omit any dependency Task with `properties.cache == True`.
            exclude_cached: Omit any dependency that is already cached in the Workspace.
            workspace: Required only if `exclude_cached=True`.

        Returns:
            A DependencyGraph containing all reachable Tasks (based on inclusion criteria).
        """
        if exclude_cached:
            workspace = resolve_auto(workspace=workspace)

        @cache
        def _include(dependency: Task) -> bool:
            """Return True if the dependency should be included."""
            if exclude_cacheable:
                return dependency.properties.cache is False
            if exclude_cached:
                return not dependency.is_cached(workspace=workspace)
            return True

        graph: DependencyGraph[Task] = DependencyGraph()
        nodes: dict[Task, int] = {}

        def _get_node(t: Task) -> int:
            """Return the graph node index for the given task."""
            i = nodes.get(t)
            if i is None:
                i = nodes[t] = graph.add_node(t)
            return i

        # DFS to traverse the Task DAG and materialize nodes/edges.
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

    def done(self, workspace: Workspace) -> bool:
        """If Workspace contains a ResultHash for this Task (expected regardless of result caching)."""
        try:
            workspace.get_result_hash(task=self)
        except RuntimeError:
            return False
        return True

    def is_running(self, workspace: Workspace) -> bool:
        """Indicator if this Task is currently running in given Workspace."""
        # For cacheable Tasks, if runtime lock (managed by Workspace) is unavailable.
        # Non-cacheable Tasks always return False, since they can freely run concurrently.
        # TODO: non-cacheable case?
        try:
            return self._runtime_lock(workspace=workspace).is_locked()
        except RuntimeError:
            # raised if dependencies are not cached (pre-requisite to Task runtime)
            return False

    def submit(
        self,
        *,
        workspace: Workspace | Literal["auto"] = "auto",
        executor: Executor | Literal["auto"] = "auto",
    ) -> DependencyGraph[Job]:
        """Submit this Task (and its dependency graph) to an Executor for deferred execution.

        Returns:
            DependencyGraph of Jobs (for monitoring progress of chunked units of work).
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
        _job_id: str | None = None,
    ) -> R:
        """Compute (or retrieve) this Task's result.

        Do minimal computation necessary to return the result. Looks up cached results whenever possible.

        Flags control which dependencies are computed:
          - compute_if_uncached: Compute if cacheable but not cached. Otherwise, this condition raises RuntimeError.
          - compute_uncached_deps: If True, (recursively) compute all uncached, cacheable dependencies.
          - job_id: Optional identifier to associate this runtime (and nested runtimes) with an executor job.

        Side effects:
            - Locking: cacheable Tasks acquire a runtime lock from the Workspace. Fails fast if lock is already held.
            - Logs: runtime stdout/stderr/logging are captured and mirrored to both stdio and the Workspace task log.
            - ResultHash: To index result object from Workspace. Task -> ResultHash mapping is stored upon completion.
            - Result: For cacheable Tasks, the computed result is stored in Workspace.

        Returns:
            Result object of function(*args, **kwargs)
        """
        workspace = resolve_auto(workspace=workspace)

        # Retrieve result if cached
        if self.properties.cache:
            if (result := workspace.results.get(self)) is not None:
                return result

            if not compute_if_uncached:
                msg = f"{self} is not cached."
                raise RuntimeError(msg)

        # Raise RuntimeError if dependencies are not cached and we don't want to compute them
        if not compute_uncached_deps and not self.are_deps_cached(workspace=workspace):
            uncached_deps = list(self.uncached_deps(workspace=workspace))
            msg = f"{self} has dependencies which must be computed and cached first: {uncached_deps}"
            raise RuntimeError(msg)

        # Get (or recursively compute) results of dependencies
        dep_results: dict[Task, Any] = {
            t: t.result(
                workspace=workspace,
                compute_if_uncached=True,
                compute_uncached_deps=True,
                _job_id=_job_id,
            )
            for t in self.dependencies
        }

        def resolve_argument(dep: Task | Any) -> Any:
            """Resolve a single argument value for execution."""
            if isinstance(dep, Task):
                return dep_results[dep]
            if dep is WORK_DIR:
                return workspace.get_work_dir(task=self)
            return dep

        # Runtime lock: should not run cached Tasks if already being computed; fails immediately if so.

        with (
            self._runtime_lock(workspace=workspace).context(blocking=False) if self.properties.cache else nullcontext()
        ):
            with workspace.open_task_log(task=self, mode="a", job_id=_job_id, timestamp="current") as task_log:
                with capture_all_output(task_log, tee_to_stdout=True):
                    args = (resolve_argument(v) for v in self.args)
                    kwargs = {k: resolve_argument(v) for k, v in self.kwargs.items()}
                    result = self.func(*args, **kwargs)

            # Store `Task -> hash(result)` mapping in Workspace
            # Indicator of Task completion

            # Index the result by the resolved Task or result object
            # "task" is better if expecting unique argument <-> result correspondence
            match self.properties.index_by:
                case "task":
                    index = self.resolved_hash(workspace=workspace)
                case "result":
                    index = result
                case _:
                    assert_never(self.properties.index_by)

            workspace.set_result_hash(self, ResultHash.from_object(index))

            # Cache result in Workspace

            if self.properties.cache:
                workspace.results[self] = result
                # TODO: if this fails, delete ResultHash?
                # Consider cases for cacheable Tasks, where ResultHash is stored, but Result is not

        return result

    def work_dir(self, workspace: Workspace | Literal["auto"] = "auto") -> Path:
        """Returns work directory from Workspace."""
        workspace = resolve_auto(workspace=workspace)
        return workspace.get_work_dir(task=self)

    def _runtime_lock(self, workspace: Workspace) -> LockLike:
        """Workspace manages a lock for each Task, expected to be held during runtime."""
        if not self.are_deps_cached(workspace=workspace):  # necessary to compute resolved_hash
            msg = f"Dependencies of {self} must be run before acquiring runtime lock"
            raise RuntimeError(msg)
        return workspace.lock(namespace="task", key=self.resolved_hash(workspace=workspace).b32())

    def __hash__(self) -> int:
        """Hash for using Task as a Pythonic key, based on `task_hash()`."""
        return hash(int(self.task_hash()))

    def __eq__(self, other: object) -> bool:
        """Task equality is based on `task_hash()`."""
        if not isinstance(other, Task):
            return NotImplemented
        return self.task_hash() == other.task_hash()

    def task_hash(self) -> TaskHash:
        """Identifier for Task, based on dependency structure."""
        # Cached as an object attribute
        # Task hash should not change, assuming Task object is not mutated (TODO)
        if not hasattr(self, "_cached_task_hash"):
            self._cached_task_hash = TaskHash.from_object((self.properties.id, self._hashed_arguments()))
        return self._cached_task_hash

    def resolved_hash(self, workspace: Workspace) -> ResolvedTaskHash:
        """Identifier for Task, based on resolved arguments. Requires dependencies to be computed first."""
        resolved_hash: ResolvedTaskHash | None = workspace.get_resolved_hash(self)

        if resolved_hash is None:
            hashed_arguments = self._hashed_arguments(hash_task_by_result=True, workspace=workspace)
            resolved_hash = ResolvedTaskHash.from_object((self.properties.id, hashed_arguments))
            workspace.set_resolved_hash(self, resolved_hash)

        return resolved_hash

    def result_hash(self, workspace: Workspace) -> ResultHash:
        """Return the stored ResultHash for this task.

        Raises:
            RuntimeError: if the task has not been computed / recorded in the workspace.
        """
        return workspace.get_result_hash(self)

    def _hashed_arguments(
        self,
        *,
        hash_task_by_result: bool = False,
        workspace: Workspace | Literal["auto"] = "auto",
    ) -> dict[str, tuple[TaskHash | ResultHash, int]]:
        """Hash for each argument.

        Args:
            hash_task_by_result: Represent Task-valued arguments by ResultHash if True, otherwise by TaskHash.
            workspace:
                For looking up a Task's ResultHash (required only if `hash_task_by_result=True`).

        Returns:
            Mapping of {argument_name : hash(argument value)}
            Excluding:
                - any keys in `properties.exclude`
                - any keys whose value equals declared default in `properties.defaults`
            Note: hash "version" of a specific (argument, value) pair can be set from `properties.versions`.
        """
        # Signature of function is used for creating {argument_name : value} map
        if not hasattr(self, "_cached_signature"):
            self._cached_signature = signature(self.func)
        bound_arguments = cast("Signature", self._cached_signature).bind(*self.args, **self.kwargs)
        bound_arguments.apply_defaults()

        # Get hash & version for each argument
        def resolve(key: str, value: Any) -> tuple[TaskHash | ResultHash, int]:
            """Return the hash and version tuple for an argument."""
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
    id: str | None = None,  # TODO: command to fill these in when None  # TODO: shadow  # noqa: A002
    cache: bool = False,
    exclude: set[str] | None = None,
    defaults: dict[str, Any] | None = None,
    versions: dict[str, dict[Any, int]] | None = None,
    index_by: Literal["task", "result"] = "result",
    serializer: type[Serializer[R]] = DefaultSerializer,  # TODO: typing
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to control how a Task is identified and cached.

    Attaches `__task_properties__: TaskProperties` attribute to `self.func`.

    Arguments:
        id:
            Stable identifier for the task definition. Will raise ValueError if None.
        cache:
            If True, `Task.result()` may store results in the Workspace.
        exclude:
            Exclude arguments (by name) from hashing.
        defaults:
            If an argument value matches the provided default, it is omitted from hashing.
        versions:
            For versioning per (argument, value) pair. Normalized to a {argument : ResultHash(value)} mapping.
        index_by:
            Determines how result is indexed (i.e. how ResultHash is computed) in Workspace:
            - "task": index by resolved task hash
            - "result": index by the result object
        serializer:
            Serializer type for saving/loading results.

    Returns:
        A decorator that mutates `func` by setting `func.__task_properties__`.
    """
    if id is None:
        msg = "id must be provided."
        raise ValueError(msg)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        """Attach task properties to the decorated function."""
        func.__task_properties__ = TaskProperties(  # ty:ignore[unresolved-attribute]
            id=id,
            cache=cache,
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
    time: int | None = None,
    nodes: int = 1,
    memory: int = 8,
    cpus: int = 1,
    gpus: int = 0,
    gpu_memory: int | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to attach resource requirements to a callable.

    Metadata used by Executors (e.g. SLURM submission) for scheduling concurrent Jobs.

    Args:
        time: Walltime (minutes)
        nodes: Number of nodes
        memory: Memory in GB
        cpus: Number of logical cores
        gpus: GPU count
        gpu_memory: Memory (GB) per GPU

    Returns:
        A decorator that mutates `func` by setting `func.__task_resources__`.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        """Attach resource requirements to the decorated function."""
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
    """Immutable metadata describing how a Task should be identified and cached."""

    id: str
    cache: bool = False
    exclude: set[str] = set()
    defaults: dict[str, Any] = {}
    versions: dict[tuple[str, ResultHash], int] = {}
    index_by: Literal["task", "result"] = "result"
    serializer: type[Serializer] = DefaultSerializer


class TaskResources(Struct):
    """Resource requirements for executing a Task. Executor-facing metadata."""

    time: int | None = None
    nodes: int = 1
    memory: int = 8
    cpus: int = 1
    gpus: int = 0
    gpu_memory: int | None = None
