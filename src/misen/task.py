"""This module provides utilities to create a Task from a Python function and arguments.

1. `Task` is used to wrap a Python function and its arguments.
2. `@task` controls how Tasks are identified, cached, and resource-parameterized.

Tasks may form a dependency graph (a DAG; see `Task.dependency_graph()`), containing edges from Tasks to dependencies.

Tasks can be submitted (`Task.submit()`) to the Executor for scheduled execution of the dependency graph.

`Task.result()` will run necessary and specified computations and return the result, retrieving and writing artifacts
(runtime logs, result, etc.) to the Workspace as needed. We typically prefer to `submit()` before calling `result()`.

Tasks can be identified by: `task_hash()` (by dependency graph) or `resolved_hash()` (by resolved arguments).
Results are identified by `result_hash()`. Any task which computes the same result should have the same `result_hash()`.
"""

from __future__ import annotations

from contextlib import nullcontext
from inspect import Signature, signature
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, Literal, ParamSpec, TypeVar, cast

from misen.utils.auto import resolve_auto
from misen.utils.frozen_mixin import FrozenMixin
from misen.utils.functions import is_function_object
from misen.utils.hashes import ResolvedTaskHash, ResultHash, TaskHash
from misen.utils.task_construction import collect_task_dependencies, hash_task_arguments
from misen.utils.task_properties import Resources, TaskProperties, resolve_task_properties
from misen.utils.task_runtime import execute_task, save_task_result

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from types import FunctionType

    from misen.executor import Executor, Job
    from misen.utils.assigned_resources import AssignedResources
    from misen.utils.graph import DependencyGraph
    from misen.utils.locks import LockLike
    from misen.workspace import Workspace

__all__ = ["Task"]

P = ParamSpec("P")
R = TypeVar("R")


class Task(FrozenMixin, Generic[R]):
    """A Task is a lazy wrapper for a function and its arguments.

    Attributes:
        func: The underlying callable.
        args / kwargs:
            Arguments to pass to `func`. Any value that is a `Task`, or contains one in nested structures,
            is considered a dependency.
        properties: TaskProperties metadata (typically attached to `func` via `@task(...)`).
        resources: Resources metadata resolved from `@task(resources=...)`.
    """

    __slots__ = (
        "_signature",
        "_task_hash",
        "args",
        "dependencies",
        "func",
        "kwargs",
        "properties",
        "resources",
    )

    def __init__(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> None:
        """Initialize a Task with a function and arguments."""
        if not is_function_object(func):
            msg = "Task func must be a Python function object."
            raise TypeError(msg)

        self.func: FunctionType = func
        self.args: P.args = args
        self.kwargs: P.kwargs = MappingProxyType(kwargs)
        self._signature: Signature = signature(func)

        self.properties: TaskProperties = resolve_task_properties(func)
        self.resources: Resources = self.properties.resources(*self.args, **self.kwargs)
        self.dependencies: frozenset[Task[Any]] = collect_task_dependencies(self.args, self.kwargs)
        self._task_hash: TaskHash = self.task_hash()

        self.freeze()

    def __repr__(self) -> str:
        """Return a short debug representation for the task."""
        return (
            f"Task({self.func.__module__}.{self.func.__qualname__}, "
            f"hash={self.task_hash().short_b32()}){' [C]' if self.properties.cache else ''}"
        )

    @property
    def T(self) -> R:  # noqa: N802
        """Cast Task to its result type. Useful for type checking compliance when passing Tasks as dependencies."""
        return cast("R", self)

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
        _assigned_resources: AssignedResources | None = None,
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

        dependency_results: dict[Task[Any], Any] = {
            dependency: dependency.result(
                workspace=workspace,
                compute_if_uncached=True,
                compute_uncached_deps=True,
                _job_id=_job_id,
                _assigned_resources=_assigned_resources,
            )
            for dependency in self.dependencies
        }

        result = execute_task(
            task=self,
            workspace=workspace,
            dependency_results=dependency_results,
            assigned_resources=_assigned_resources,
            job_id=_job_id,
            lock_context=(
                self._runtime_lock(workspace=workspace).context(blocking=False)
                if self.properties.cache
                else nullcontext()
            ),
        )

        save_task_result(task=self, result=result, workspace=workspace)

        return result

    def work_dir(self, workspace: Workspace | Literal["auto"] = "auto") -> Path:
        """Returns work directory from Workspace."""
        workspace = resolve_auto(workspace=workspace)
        return workspace.get_work_dir(task=self)

    def task_hash(self) -> TaskHash:
        """Identifier for Task, based on dependency structure."""
        if hasattr(self, "_task_hash"):
            return self._task_hash
        hashed_args = hash_task_arguments(
            signature=self._signature,
            args=self.args,
            kwargs=self.kwargs,
            properties=self.properties,
        )
        return TaskHash.from_object((self.properties.id, hashed_args))

    def resolved_hash(self, workspace: Workspace) -> ResolvedTaskHash:
        """Identifier for Task, based on resolved arguments. Requires dependencies to be computed first."""
        resolved_hash: ResolvedTaskHash | None = workspace.get_resolved_hash(self)

        if resolved_hash is None:
            hashed_args = hash_task_arguments(
                signature=self._signature,
                args=self.args,
                kwargs=self.kwargs,
                properties=self.properties,
                hash_task_by_result=True,
                workspace=workspace,
            )
            resolved_hash = ResolvedTaskHash.from_object((self.properties.id, hashed_args))
            workspace.set_resolved_hash(self, resolved_hash)

        return resolved_hash

    def result_hash(self, workspace: Workspace) -> ResultHash:
        """Return the stored ResultHash for this task.

        Raises:
            RuntimeError: if the task has not been computed / recorded in the workspace.
        """
        return workspace.get_result_hash(self)

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
