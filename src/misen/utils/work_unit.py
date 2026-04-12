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

from misen.utils.nested import map_nested_leaves
from misen.utils.task_utils import build_task_dependency_graph

if TYPE_CHECKING:
    from misen.task_metadata import GpuRuntime, Resources
    from misen.tasks import Task
    from misen.utils.assigned_resources import AssignedResources, AssignedResourcesPerNode
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
    tasks (max for CPU/memory/GPU counts; sum for finite runtime).
    """

    __slots__ = ("dependencies", "graph", "resources", "root")

    root: Task
    graph: DependencyGraph[Task]
    resources: Resources
    dependencies: set[WorkUnit]

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
        from misen.task_metadata import Resources

        resource_list: list[Resources] = [task.resources for task in self.graph.nodes()]
        gpu_runtimes = cast(
            "set[GpuRuntime]",
            {resource.gpu_runtime for resource in resource_list if resource.gpus > 0},
        )
        match len(gpu_runtimes):
            case 0:
                gpu_runtime = "cuda"
            case 1:
                (gpu_runtime,) = gpu_runtimes
            case _:
                msg = f"WorkUnit has incompatible gpu_runtime requirements: {gpu_runtimes}"
                raise ValueError(msg)

        self.resources = Resources(
            time=(
                None
                if any(resource.time is None for resource in resource_list)
                else sum(cast("int", resource.time) for resource in resource_list)
            ),
            nodes=max(resource.nodes for resource in resource_list),
            memory=max(resource.memory for resource in resource_list),
            cpus=max(resource.cpus for resource in resource_list),
            gpus=max(resource.gpus for resource in resource_list),
            gpu_memory=(
                None
                if all(resource.gpu_memory is None for resource in resource_list)
                else max(resource.gpu_memory for resource in resource_list if resource.gpu_memory is not None)
            ),
            gpu_runtime=gpu_runtime,
        )

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
        assigned_resources: AssignedResources | AssignedResourcesPerNode | None,
    ) -> None:
        """Execute tasks in dependency order for a task graph.

        Args:
            graph: Task dependency graph to execute.
            workspace: Workspace used for cache/log/storage operations.
            job_id: Job identifier propagated into task log naming.
            assigned_resources: Runtime-assigned resources for sentinel injection.
        """
        from misen.tasks import Task

        task_results: dict[Task[Any], Any] = {}

        def resolve_arg(arg: Any) -> Any:
            """Resolve non-cacheable task leaves from in-memory runtime results."""

            def resolve_leaf(leaf: Any) -> Any:
                if isinstance(leaf, Task) and not leaf.properties.cache:
                    return task_results[leaf]
                return leaf

            return map_nested_leaves(arg, resolve_leaf)

        ordered_tasks: list[Task[Any]] = list(graph)
        for i, task in enumerate(ordered_tasks):
            # Keep only transient results still needed by future in-unit tasks.
            remaining_deps = {
                dependency for remaining_task in ordered_tasks[i:] for dependency in remaining_task.dependencies
            }
            task_results = {k: v for k, v in task_results.items() if k in remaining_deps}

            # Rebuild the task with resolved in-unit non-cacheable dependencies.
            # Cacheable dependencies are still loaded through Workspace in Task.result.
            task_results[task] = task._with_resolved_args(
                args=tuple(resolve_arg(arg) for arg in task.args),
                kwargs={name: resolve_arg(arg) for name, arg in task.kwargs.items()},
            ).result(
                workspace=workspace,
                compute_if_uncached=True,
                compute_uncached_deps=False,
                _job_id=job_id,
                _assigned_resources=assigned_resources,
            )

    def as_payload(self, workspace: Workspace, job_id: str) -> bytes:
        """Serialize executable payload for backend dispatch.

        Args:
            workspace: Workspace instance captured in payload closure.
            job_id: Job id captured for logging.
            assigned_resources_getter: Resource getter captured for runtime
                sentinel resolution.

        Returns:
            Cloudpickle payload bytes.
        """
        return cloudpickle.dumps(
            functools.partial(
                WorkUnit.execute,
                graph=self.graph,
                workspace=workspace,
                job_id=job_id,
            )
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
    anchor_graph = task_graph.copy()
    anchors = [i for i in anchor_graph.node_indices() if anchor_graph.is_root(i) or anchor_graph[i].properties.cache]
    anchor_graph.coarsen_to_anchors(anchors=anchors)

    # Materialize WorkUnit nodes preserving dependency topology.
    work_graph = cast("DependencyGraph[WorkUnit]", anchor_graph.copy())
    for i in work_graph.evaluation_order():
        work_graph[i] = WorkUnit(root=anchor_graph[i], dependencies=set(work_graph.successors(i)))

    return work_graph
