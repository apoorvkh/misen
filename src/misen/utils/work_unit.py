"""Work-unit decomposition for cache-aware scheduling.

``WorkUnit`` is the bridge between task-level semantics and executor-level
concurrency. A work unit groups a connected subgraph of non-cacheable tasks and
cuts edges at cacheable boundaries, so backends can schedule coarse units while
preserving cache semantics.
"""

from __future__ import annotations

import functools
from operator import is_
from typing import TYPE_CHECKING, Any, cast

import cloudpickle

from misen.sentinels import DASK_CLIENT
from misen.task_metadata import Resources, aggregate_resources
from misen.utils.nested import map_nested_leaves
from misen.utils.runtime_values import RuntimeValues
from misen.utils.task_utils import build_task_dependency_graph

if TYPE_CHECKING:
    from collections.abc import Mapping

    from misen.tasks import Task
    from misen.utils.graph import DependencyGraph
    from misen.workspace import Workspace

__all__ = ["WorkUnit", "build_work_graph"]


class WorkUnit:
    """Cache-bounded unit of execution derived from a task DAG.

    A WorkUnit contains the non-cacheable subgraph rooted at ``root`` and
    bounded by downstream cacheable nodes. Those boundary nodes become separate
    WorkUnits referenced via ``dependencies``.

    Tasks inside a WorkUnit execute sequentially in dependency order.
    Scheduler-facing resources are aggregated conservatively across contained
    tasks (max for node/CPU/memory/accelerator counts; sum for finite runtime).
    """

    __slots__ = ("dependencies", "graph", "resources", "root", "uses_dask_client")

    root: Task
    graph: DependencyGraph[Task]
    resources: Resources
    dependencies: set[WorkUnit]
    uses_dask_client: bool

    def __init__(self, root: Task, dependencies: set[WorkUnit]) -> None:
        """Initialize a WorkUnit rooted at the given task.

        Args:
            root: Cacheable root task for this work unit.
            dependencies: Downstream work units that depend on this unit.
        """
        self.root = root
        self.dependencies = dependencies

        # Exclude downstream cacheable tasks: they are represented as dependent
        # WorkUnits instead of in-unit tasks.
        self.graph = build_task_dependency_graph(task=root, exclude_cacheable=True)

        # Compute one scheduler request that satisfies every task in the unit.
        tasks = self.graph.nodes()
        self.resources = aggregate_resources(task.resources for task in tasks)
        dask_topologies = {
            _dask_topology(task.resources)
            for task in tasks
            if task._requests_runtime_value(DASK_CLIENT)  # noqa: SLF001
        }
        self.uses_dask_client = bool(dask_topologies)
        if any(nodes == 1 for nodes, *_ in dask_topologies):
            msg = "DASK_CLIENT requires nodes > 1."
            raise ValueError(msg)
        allocation = _dask_topology(self.resources)
        if dask_topologies and dask_topologies != {allocation}:
            msg = f"DASK_CLIENT tasks must request the WorkUnit's exact topology {allocation!r}."
            raise ValueError(msg)

    def __hash__(self) -> int:
        """Return hash keyed by root task identity."""
        return hash(self.root)

    def __eq__(self, other: object) -> bool:
        """Return equality based on root task identity."""
        return isinstance(other, WorkUnit) and self.root == other.root

    def __repr__(self) -> str:
        """Return compact debug representation."""
        root_repr = repr(self.root).removesuffix(" [C]")
        if root_repr.startswith("Task(") and root_repr.endswith(")"):
            root_repr = root_repr[len("Task(") : -1]
        return f"WorkUnit({root_repr})"

    def done(self, workspace: Workspace) -> bool:
        """Return whether work unit has been completed."""
        return self.root.done(workspace=workspace)

    @staticmethod
    def execute(
        graph: DependencyGraph[Task[Any]],
        workspace: Workspace,
        job_id: str,
    ) -> None:
        """Execute tasks in dependency order for a task graph.

        Args:
            graph: Task dependency graph to execute.
            workspace: Workspace used for cache/log/storage operations.
            job_id: Job identifier propagated into task log naming.
        """
        from misen.tasks import Task

        task_results: dict[Task[Any], Any] = {}

        def resolve_leaf(value: Any) -> Any:
            """Resolve a non-cacheable task from the in-memory result map."""
            return task_results[value] if isinstance(value, Task) and not value.meta.cache else value

        with RuntimeValues() as runtime_values:
            ordered_tasks: list[Task[Any]] = list(graph)
            last_use = {
                dependency: index
                for index, task in enumerate(ordered_tasks)
                for dependency in task.dependencies
                if not dependency.meta.cache
            }
            for i, task in enumerate(ordered_tasks):
                # Rebuild the task with resolved in-unit non-cacheable dependencies.
                # Cacheable dependencies are still loaded through Workspace in Task.result.
                executable_task = task.with_resolved_args(
                    args=tuple(map_nested_leaves(arg, resolve_leaf) for arg in task.args),
                    kwargs={name: map_nested_leaves(arg, resolve_leaf) for name, arg in task.kwargs.items()},
                )
                result = executable_task.result(
                    workspace=workspace,
                    compute_if_uncached=True,
                    compute_uncached_deps=False,
                    _job_id=job_id,
                    _log_task=task,
                    _runtime_values=runtime_values,
                )
                if task in last_use:
                    task_results[task] = result
                del result
                for dependency in task.dependencies:
                    if last_use.get(dependency) == i:
                        task_results.pop(dependency)

    def as_payload(
        self,
        workspace: Workspace,
        job_id: str,
        *,
        submission_id: str | None = None,
        dependency_jobs: Mapping[WorkUnit, str] | None = None,
    ) -> bytes:
        """Serialize executable payload for backend dispatch.

        The payload bundles the workspace separately from the callable so
        the worker entrypoint can wrap the call in
        :meth:`Workspace.streaming_job_log` without unpacking a closure.

        Args:
            workspace: Workspace instance, surfaced for the worker's
                streaming-log context.
            job_id: Job id captured for logging.
            submission_id: Submission namespace for dependency markers.
            dependency_jobs: Prerequisite work units mapped to their Misen
                job ids. ``None`` disables the dependency gate.

        Returns:
            Cloudpickle payload bytes containing ``{"workspace": ..., "fn": ...}``
            where ``fn`` is a zero-arg callable.
        """
        execute = functools.partial(
            WorkUnit.execute,
            graph=self.graph,
            workspace=workspace,
            job_id=job_id,
        )
        if dependency_jobs is not None:
            if submission_id is None:
                msg = "submission_id is required when dependency_jobs are provided."
                raise ValueError(msg)
            from misen.utils.job_dependencies import run_with_dependencies
            from misen.utils.runtime_events import work_unit_label

            execute = functools.partial(
                run_with_dependencies,
                execute,
                workspace=workspace,
                submission_id=submission_id,
                job_id=job_id,
                dependencies=tuple(
                    (dependency_id, work_unit_label(dependency))
                    for dependency, dependency_id in dependency_jobs.items()
                ),
            )
        return cloudpickle.dumps(
            {
                "workspace": workspace,
                "fn": execute,
            }
        )


def build_work_graph(tasks: set[Task]) -> DependencyGraph[WorkUnit]:
    """Transform task DAG into work-unit DAG.

    Args:
        tasks: Root tasks requested for execution.

    Returns:
        Dependency graph of work units ready for executor submission.
    """
    from misen.tasks import Task

    # Edge convention: A -> B means A depends on B.
    union = Task((lambda *_: None), *tasks)
    task_graph: DependencyGraph[Task[Any]] = build_task_dependency_graph(task=union)
    task_graph.remove_node_by_value(union, cmp=is_, first=True)

    # Keep only roots and cache boundaries, then retain induced connectivity.
    anchors = [i for i in task_graph.node_indices() if task_graph.is_root(i) or task_graph[i].meta.cache]
    task_graph.coarsen_to_anchors(anchors=anchors)

    # Materialize WorkUnit nodes in place; dependencies have already been converted.
    work_graph = cast("DependencyGraph[WorkUnit]", task_graph)
    for i in work_graph.evaluation_order():
        work_graph[i] = WorkUnit(root=cast("Task[Any]", work_graph[i]), dependencies=set(work_graph.successors(i)))

    return work_graph


def _dask_topology(resources: Resources) -> tuple[int, int, str | None]:
    """Return the allocation-shaping subset of a resource request."""
    accelerators = resources["accelerators"]
    return resources["nodes"], accelerators, resources["accelerator_type"] if accelerators else None
