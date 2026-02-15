"""Work-unit decomposition for cache-aware scheduling.

``WorkUnit`` is the bridge between task-level semantics and executor-level
concurrency. A work unit groups a connected subgraph of non-cacheable tasks and
cuts edges at cacheable boundaries, so backends can schedule coarse units while
preserving cache semantics.
"""

from __future__ import annotations

import functools
from functools import cache
from operator import is_
from typing import TYPE_CHECKING, Any, cast

import cloudpickle

from misen.tasks import Task
from misen.utils.graph import DependencyGraph
from misen.utils.task_utils import map_nested_leaves

if TYPE_CHECKING:
    from collections.abc import Callable

    from misen.task_properties import Resources
    from misen.utils.assigned_resources import AssignedResources
    from misen.workspace import Workspace


class WorkUnit:
    """Cache-bounded unit of execution derived from a task DAG.

    It corresponds to the sub-DAG of non-cacheable tasks reachable from `root`, truncated at downstream cacheable tasks
    (which become separate WorkUnits, i.e. `dependencies`).

    Execution: Tasks in `graph` are executed sequentially in dependency order via `execute()`.

    Specifies minimum resources to execute any Task in the DAG.
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

        # Dependency graph of tasks.
        # Truncated at caching boundaries since downstream cacheable nodes become WorkUnit dependencies.
        self.graph = _build_task_dependency_graph(task=root, exclude_cacheable=True)

        # Union of resources for all tasks in graph
        from misen.task_properties import Resources

        _resource_list: list[Resources] = [task.resources for task in self.graph.nodes()]
        self.resources = Resources(
            time=(
                None
                if any(r.time is None for r in _resource_list)
                else sum(cast("int", r.time) for r in _resource_list)
            ),
            nodes=max(r.nodes for r in _resource_list),
            memory=max(r.memory for r in _resource_list),
            cpus=max(r.cpus for r in _resource_list),
            gpus=max(r.gpus for r in _resource_list),
            gpu_memory=(
                None
                if all(r.gpu_memory is None for r in _resource_list)
                else max(r.gpu_memory for r in _resource_list if r.gpu_memory is not None)
            ),
        )

    def __hash__(self) -> int:
        """Return hash keyed by root task identity."""
        return hash(self.root)

    def __eq__(self, other: object) -> bool:
        """Return equality based on root task identity."""
        return isinstance(other, WorkUnit) and self.root == other.root

    def __repr__(self) -> str:
        """Return compact debug representation."""
        return f"WorkUnit(hash={self.root.task_hash().short_b32()})"

    def execute(
        self,
        workspace: Workspace,
        job_id: str,
        assigned_resources_getter: Callable[[], AssignedResources | None],
    ) -> None:
        """Execute tasks in dependency order inside this work unit.

        Args:
            workspace: Workspace used for cache/log/storage operations.
            job_id: Job identifier propagated into task log naming.
            assigned_resources_getter: Callable returning runtime-assigned
                resources for sentinel injection.
        """
        from misen.tasks import Task

        task_results: dict[Task, Any] = {}
        assigned_resources = assigned_resources_getter()

        def resolve_arg(arg: Any) -> Any:
            """Resolve non-cacheable task leaves from in-memory runtime results."""

            def resolve_leaf(leaf: Any) -> Any:
                if isinstance(leaf, Task) and not leaf.properties.cache:
                    return task_results[leaf]
                return leaf

            return map_nested_leaves(arg, resolve_leaf)

        ordered_tasks: list[Task] = list(self.graph)
        for i, task in enumerate(ordered_tasks):
            # only keep task results needed by future dependents
            remaining_deps = {d for t in ordered_tasks[i:] for d in t.dependencies}
            task_results = {k: v for k, v in task_results.items() if k in remaining_deps}

            # execute task
            # retrieving results of non-cacheable dependencies from `task_results`
            # cacheable dependencies will be resolved from Workspace in `.result()`
            task_results[task] = Task(
                task.func,
                *(resolve_arg(dep) for dep in task.args),
                **{k: resolve_arg(dep) for k, dep in task.kwargs.items()},
            ).result(
                workspace=workspace,
                compute_if_uncached=True,
                compute_uncached_deps=False,
                _job_id=job_id,
                _assigned_resources=assigned_resources,
            )

    def as_payload(
        self,
        workspace: Workspace,
        job_id: str,
        assigned_resources_getter: Callable[[], AssignedResources | None],
    ) -> bytes:
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
                self.execute,
                workspace=workspace,
                job_id=job_id,
                assigned_resources_getter=assigned_resources_getter,
            )
        )


def build_work_graph(tasks: set[Task]) -> DependencyGraph[WorkUnit]:
    """Transform task DAG into work-unit DAG.

    Args:
        tasks: Root tasks requested for execution.

    Returns:
        Dependency graph of work units ready for executor submission.
    """
    # Task dependency graph: an edge from A to B means A depends on B
    union = Task((lambda *_: None), *tasks)
    task_graph: DependencyGraph[Task] = _build_task_dependency_graph(task=union)
    task_graph.remove_node_by_value(union, cmp=is_, first=True)

    # Retain only root and cachable tasks (and the induced graph minor)
    anchor_graph = task_graph.copy()
    anchors = [i for i in anchor_graph.node_indices() if anchor_graph.is_root(i) or anchor_graph[i].properties.cache]
    anchor_graph.coarsen_to_anchors(anchors=anchors)

    # replace nodes with WorkUnit instances
    work_graph = cast("DependencyGraph[WorkUnit]", anchor_graph.copy())
    for i in work_graph.evaluation_order():
        work_graph[i] = WorkUnit(root=anchor_graph[i], dependencies=set(work_graph.successors(i)))

    return work_graph


def _build_task_dependency_graph(
    task: Task[Any],
    *,
    exclude_cacheable: bool = False,
    exclude_cached: bool = False,
    workspace: Workspace | None = None,
) -> DependencyGraph[Task[Any]]:
    """Build dependency graph rooted at a task-like object.

    Args:
        task: Root task-like node.
        exclude_cacheable: Whether to skip cacheable dependency nodes.
        exclude_cached: Whether to skip dependencies already cached in workspace.
        workspace: Workspace required when ``exclude_cached=True``.

    Returns:
        Dependency graph with edges ``task -> dependency``.

    Raises:
        ValueError: If ``exclude_cached=True`` and workspace is not provided.
    """
    if exclude_cacheable:

        @cache
        def include_dependency(dependency: Task[Any]) -> bool:
            return dependency.properties.cache is False

    elif exclude_cached:
        if workspace is None:
            msg = "workspace is required when exclude_cached=True."
            raise ValueError(msg)

        @cache
        def include_dependency(dependency: Task[Any]) -> bool:
            return not dependency.is_cached(workspace=workspace)

    else:

        def include_dependency(dependency: Task[Any]) -> bool:  # noqa: ARG001
            return True

    graph: DependencyGraph[Task[Any]] = DependencyGraph()
    nodes: dict[Task[Any], int] = {}

    def get_node_index(candidate: Task[Any]) -> int:
        node_index = nodes.get(candidate)
        if node_index is None:
            node_index = nodes[candidate] = graph.add_node(candidate)
        return node_index

    stack: list[Task[Any]] = [task]
    seen: set[Task[Any]] = {task}

    while stack:
        current = stack.pop()
        current_node = get_node_index(current)

        for dependency in current.dependencies:
            if not include_dependency(dependency):
                continue
            graph.add_edge(current_node, get_node_index(dependency), None)
            if dependency not in seen:
                seen.add(dependency)
                stack.append(dependency)

    return graph
