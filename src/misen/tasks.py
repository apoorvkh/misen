"""Task abstraction: lazy DAG nodes with deterministic identity.

Core design decisions:

- ``Task`` wraps a Python function plus bound arguments and is itself immutable.
- Any nested ``Task`` argument is treated as a dependency, producing a DAG.
- Identity is split into:
  - ``task_hash``: structural identity (before dependency resolution)
  - ``resolved_hash``: identity after dependency results are known
  - ``result_hash``: identity of function result when explicitly hashable,
    otherwise resolved task identity
- Runtime execution is workspace-driven so caching and locking semantics are
  consistent across local and distributed executors.
- For a fixed workspace, cacheable tasks execute under a per-task runtime lock
  (single active instance), while non-cacheable tasks do not take this lock and
  may execute concurrently.

``Task.result()`` supports eager local execution; ``Task.submit()`` routes to an
executor for dependency-aware concurrent scheduling.
"""

from __future__ import annotations

import itertools
import logging
import shutil
from contextlib import nullcontext
from inspect import Signature, signature
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, Literal, ParamSpec, TypeVar, cast

from misen.sentinels import ASSIGNED_RESOURCES, ASSIGNED_RESOURCES_PER_NODE
from misen.task_metadata import Resources, TaskMetadata, resolve_task_metadata
from misen.utils.frozen_mixin import FrozenMixin
from misen.utils.function_introspection import is_function_object
from misen.utils.hashing import ResolvedTaskHash, ResultHash, TaskHash
from misen.utils.task_utils import collect_task_dependencies, execute_task, hash_task_arguments, save_task_result

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path
    from types import FunctionType

    from misen.executor import Executor, Job
    from misen.utils.assigned_resources import AssignedResources, AssignedResourcesPerNode
    from misen.utils.graph import DependencyGraph
    from misen.utils.locks import LockLike
    from misen.workspace import Workspace

__all__ = ["Task"]

P = ParamSpec("P")
R = TypeVar("R")
TRACE_LEVEL = logging.DEBUG - 5
logger = logging.getLogger(__name__)


class Task(FrozenMixin, Generic[R]):
    """A Task is a lazy wrapper for a function and its arguments.

    Attributes:
        func: Underlying callable.
        args: Positional arguments for ``func``.
        kwargs: Keyword arguments for ``func``.
        properties: Metadata resolved from :func:`misen.task_metadata.meta`.
        resources: Runtime resource request derived from bound arguments.
        dependencies: Immediate dependent tasks discovered from nested args.
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
        """Initialize a task from a Python function and bound arguments.

        Args:
            func: Python function object to wrap.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Raises:
            TypeError: If ``func`` is not a Python function object.
        """
        if not is_function_object(func):
            msg = "Task func must be a Python function object."
            raise TypeError(msg)

        self.func: FunctionType = func
        self.args: P.args = args
        self.kwargs: Mapping[str, Any] = MappingProxyType(kwargs)
        self._signature: Signature = signature(func)

        self.properties: TaskMetadata = resolve_task_metadata(func)
        self.resources: Resources = self.properties.resources(*self.args, **self.kwargs)

        values = tuple(itertools.chain(self.args, self.kwargs.values()))

        if self.resources.nodes > 1 and ASSIGNED_RESOURCES in values:
            msg = "ASSIGNED_RESOURCES cannot be used when resources.nodes > 1; use ASSIGNED_RESOURCES_PER_NODE."
            raise ValueError(msg)
        if self.resources.nodes == 1 and ASSIGNED_RESOURCES_PER_NODE in values:
            msg = "ASSIGNED_RESOURCES_PER_NODE cannot be used when resources.nodes == 1; use ASSIGNED_RESOURCES."
            raise ValueError(msg)

        self.dependencies: frozenset[Task[Any]] = collect_task_dependencies(self.args, self.kwargs)
        self._task_hash: TaskHash = self.task_hash()

        self.freeze()

    def __repr__(self) -> str:
        """Return compact debug representation."""
        argument_items = self._repr_argument_items()
        label = self.properties.id
        if argument_items:
            label = f"{label}({', '.join(argument_items)})"
        task_hash = self.task_hash()
        label = f"{label} [{task_hash.short_b32()}]"
        return f"Task({label}){' [C]' if self.properties.cache else ''}"

    @staticmethod
    def _contains_task_reference(value: Any) -> bool:
        """Return whether a value contains nested task references."""
        if isinstance(value, Task):
            return True
        if type(value) is dict:
            return any(Task._contains_task_reference(k) for k in value) or any(
                Task._contains_task_reference(v) for v in value.values()
            )
        if type(value) in {list, tuple, set, frozenset}:
            return any(Task._contains_task_reference(v) for v in value)
        return False

    def _repr_argument_items(self) -> list[str]:
        """Return argument fragments included in ``Task.__repr__``."""
        bound = self._signature.bind_partial(*self.args, **self.kwargs)
        items: list[str] = []

        for name, value in bound.arguments.items():
            if name in self.properties.exclude:
                continue
            if Task._contains_task_reference(value):
                continue
            items.append(f"{name}={value!r}")

        return items

    @property
    def T(self) -> R:  # noqa: N802
        """Type-only cast to the task's result type.

        Returns:
            ``self`` cast to ``R`` for static typing in dependency wiring.
        """
        return cast("R", self)

    def is_cached(self, workspace: Workspace | Literal["auto"] = "auto") -> bool:
        """Return whether this task has a cached result.

        Args:
            workspace: Workspace instance or ``"auto"``.

        Returns:
            ``True`` if this task is cacheable and present in workspace cache.
        """
        from misen.workspace import Workspace

        workspace = Workspace.resolve_auto(workspace)
        return self.properties.cache and self in workspace.results

    def uncached_deps(self, workspace: Workspace | Literal["auto"] = "auto") -> Iterator[Task]:
        """Yield immediate dependencies that are cacheable but currently missing.

        Args:
            workspace: Workspace instance or ``"auto"``.

        Yields:
            Immediate dependency tasks that need computation.
        """
        return (
            dependency
            for dependency in self.dependencies
            if dependency.properties.cache and not dependency.is_cached(workspace=workspace)
        )

    def are_deps_cached(self, workspace: Workspace | Literal["auto"] = "auto") -> bool:
        """Return whether all immediate cacheable dependencies are available.

        Args:
            workspace: Workspace instance or ``"auto"``.

        Returns:
            ``True`` when no immediate cacheable dependency is missing.
        """
        return not any(self.uncached_deps(workspace=workspace))

    def done(self, workspace: Workspace) -> bool:
        """Return whether this task is complete in the workspace.

        For non-cacheable tasks (``cache=False``), completion means a result
        hash has been recorded.  For cacheable tasks (``cache=True``),
        completion additionally requires the result payload to be present.

        Args:
            workspace: Workspace to query.

        Returns:
            ``True`` if the task is complete per its caching policy.
        """
        try:
            workspace.get_result_hash(task=self)
        except RuntimeError:
            return False
        if self.properties.cache and self not in workspace.results:
            return False
        return True

    def is_running(self, workspace: Workspace) -> bool:
        """Return whether this task currently holds its runtime lock.

        Args:
            workspace: Workspace providing task locks.

        Returns:
            ``True`` when the cacheable task runtime lock is held.

        Notes:
            Non-cacheable tasks are never represented as "running" here because
            they do not acquire workspace runtime locks.
        """
        try:
            return self._runtime_lock(workspace=workspace).is_locked()
        except RuntimeError:
            # Raised when dependencies are unresolved and therefore the runtime
            # lock key (resolved hash) cannot be computed yet.
            return False

    def result(
        self,
        *,
        workspace: Workspace | Literal["auto"] = "auto",
        compute_if_uncached: bool = False,
        compute_uncached_deps: bool = False,
        _job_id: str | None = None,
        _assigned_resources: AssignedResources | AssignedResourcesPerNode | None = None,
    ) -> R:
        """Compute (or retrieve) this Task's result.

        Args:
            workspace: Workspace instance or ``"auto"``.
            compute_if_uncached: Whether to compute this task when its cached
                value is missing.
            compute_uncached_deps: Whether to recursively compute uncached
                dependencies.
            _job_id: Optional executor job identifier for log grouping.
            _assigned_resources: Optional runtime resources injected by executor.

        Returns:
            Result of ``func(*resolved_args, **resolved_kwargs)``.

        Raises:
            RuntimeError: If required cache entries are missing and computation
                flags do not permit executing missing nodes.

        Notes:
            Cacheable tasks acquire a workspace runtime lock before execution.
            For a given workspace/resolved task identity, this enforces a
            single active runtime. Non-cacheable tasks skip runtime locking and
            can run concurrently.
            Upon successful completion, non-cacheable task work dirs are
            cleaned up; cacheable task work dirs are cleaned up when
            ``@meta(cleanup_work_dir=True)`` is set.
            Logs are captured to task logs and optionally mirrored to stdout.
        """
        from misen.workspace import Workspace

        workspace = Workspace.resolve_auto(workspace)
        logger.debug(
            "Resolving task result for %s (cache=%s, compute_if_uncached=%s, compute_uncached_deps=%s, job_id=%s).",
            self,
            self.properties.cache,
            compute_if_uncached,
            compute_uncached_deps,
            _job_id or "n/a",
        )

        # Fast path: return cached payload for cacheable tasks.
        if self.properties.cache:
            if (result := workspace.results.get(self)) is not None:
                logger.log(TRACE_LEVEL, "Cache hit for %s.", self)
                return result

            logger.log(TRACE_LEVEL, "Cache miss for %s.", self)
            if not compute_if_uncached:
                msg = f"{self} is not cached."
                logger.warning("Task result requested without compute_if_uncached and cache is missing for %s.", self)
                raise RuntimeError(msg)

        # Guardrail: only recurse into dependencies when explicitly requested.
        if not compute_uncached_deps and not self.are_deps_cached(workspace=workspace):
            uncached_deps = list(self.uncached_deps(workspace=workspace))
            logger.debug(
                "Task %s has %d uncached dependency(ies) while compute_uncached_deps is disabled.",
                self,
                len(uncached_deps),
            )
            msg = f"{self} has dependencies which must be computed and cached first: {uncached_deps}"
            raise RuntimeError(msg)

        if self.dependencies:
            logger.debug("Resolving %d dependency result(s) for %s.", len(self.dependencies), self)
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

        lock_context = (
            self._runtime_lock(workspace=workspace).context(blocking=False) if self.properties.cache else nullcontext()
        )

        if self.properties.cache:
            logger.debug("Acquiring runtime lock for %s.", self)
        with lock_context:
            result, work_dir = execute_task(
                task=self,
                workspace=workspace,
                dependency_results=dependency_results,
                assigned_resources=_assigned_resources,
                job_id=_job_id,
            )

        save_task_result(task=self, result=result, workspace=workspace)
        logger.debug("Persisted task result metadata for %s.", self)
        should_cleanup = not self.properties.cache or self.properties.cleanup_work_dir
        if should_cleanup and work_dir is not None and work_dir.exists():
            shutil.rmtree(work_dir)
            logger.debug("Removed work directory for %s at %s.", self, work_dir)

        return result

    def submit(
        self,
        *,
        workspace: Workspace | Literal["auto"] = "auto",
        executor: Executor | Literal["auto"] = "auto",
        blocking: bool = False,
    ) -> DependencyGraph[Job]:
        """Submit this task DAG to an executor for deferred execution.

        Args:
            workspace: Workspace instance or ``"auto"``.
            executor: Executor instance or ``"auto"``.
            blocking: Whether to wait for all submitted jobs to terminate
                before returning.

        Returns:
            Dependency graph of job handles for scheduled work units.
        """
        from misen.executor import Executor
        from misen.workspace import Workspace

        executor = Executor.resolve_auto(executor)
        workspace = Workspace.resolve_auto(workspace)
        job_graph, _snapshot = executor.submit(tasks={self}, workspace=workspace, blocking=blocking)
        return job_graph

    def work_dir(self, workspace: Workspace | Literal["auto"] = "auto") -> Path:
        """Return this task's working directory.

        Args:
            workspace: Workspace instance or ``"auto"``.

        Returns:
            Per-task workspace directory.
        """
        from misen.workspace import Workspace

        workspace = Workspace.resolve_auto(workspace)
        return workspace.get_work_dir(task=self)

    def task_hash(self) -> TaskHash:
        """Return structural hash for this task.

        Returns:
            Hash derived from task id plus argument structure, where dependency
            leaves are represented by dependency task hashes and non-Task
            values must hash through explicit ``stable_hash`` handlers.
        """
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
        """Return resolved-input hash for this task.

        Args:
            workspace: Workspace used to resolve dependency result hashes.

        Returns:
            Hash derived from task id plus argument values after dependency
            resolution, where dependency leaves are represented by dependency
            result hashes.
        """
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

        Args:
            workspace: Workspace holding task completion metadata.

        Returns:
            Result hash recorded for this task execution. This indexes by the
            result value when it has an explicit stable-hash handler,
            otherwise by the resolved task identity.

        Raises:
            RuntimeError: if the task has not been computed / recorded in the workspace.
        """
        return workspace.get_result_hash(self)

    def _runtime_lock(self, workspace: Workspace) -> LockLike:
        """Return workspace runtime lock for this task.

        Args:
            workspace: Workspace lock provider.

        Returns:
            Lock-like object keyed by resolved hash.

        Raises:
            RuntimeError: If dependencies are not cached yet.

        Notes:
            This lock is used only for cacheable tasks. It enforces that, for a
            specific workspace and resolved task identity, only one runtime is
            active at a time.
        """
        if not self.are_deps_cached(workspace=workspace):
            msg = f"Dependencies of {self} must be run before acquiring runtime lock"
            raise RuntimeError(msg)
        return workspace.lock(namespace="task", key=self.resolved_hash(workspace=workspace).b32())

    def __hash__(self) -> int:
        """Return Python hash based on :meth:`task_hash`."""
        return hash(int(self.task_hash()))

    def __eq__(self, other: object) -> bool:
        """Return equality based on :meth:`task_hash`."""
        if not isinstance(other, Task):
            return NotImplemented
        return self.task_hash() == other.task_hash()
