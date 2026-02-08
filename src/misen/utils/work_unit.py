"""Work unit definition as a cache-bounded DAG of Tasks."""

from __future__ import annotations

import functools
from operator import is_
from typing import TYPE_CHECKING, Any, cast

import cloudpickle

from misen.task import Task
from misen.utils.hashes import short_hash

if TYPE_CHECKING:
    from misen.task import TaskResources
    from misen.utils.graph import DependencyGraph
    from misen.workspace import Workspace


class WorkUnit:
    """A WorkUnit is a cache-bounded unit of execution.

    It corresponds to the sub-DAG of non-cacheable tasks reachable from `root`, truncated at downstream cacheable tasks
    (which become separate WorkUnits, i.e. `dependencies`).

    Execution: Tasks in `graph` are executed sequentially in dependency order via `execute()`.

    Specifies minimum resources to execute any Task in the DAG.
    """

    __slots__ = ("dependencies", "graph", "resources", "root")

    root: Task
    graph: DependencyGraph[Task]
    resources: TaskResources
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
        self.graph = root.dependency_graph(exclude_cacheable=True)

        # Union of resources for all tasks in graph
        from misen.task import TaskResources

        _resource_list: list[TaskResources] = [task.resources for task in self.graph.nodes()]
        self.resources = TaskResources(
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
        """WorkUnits have a 1:1 correspondence with the `root` Task, so we can defer to hash(`root`)."""
        return hash(self.root)

    def __eq__(self, other: object) -> bool:
        """Return True if the other WorkUnit has the same root task."""
        return isinstance(other, WorkUnit) and self.root == other.root

    def __repr__(self) -> str:
        """Return a short debug representation for the work unit."""
        return f"WorkUnit(hash={short_hash(self)})"

    def execute(self, workspace: Workspace) -> None:
        """Execute self.graph Tasks one-by-one in dependency order. Should be called by `Executor._dispatch()`."""
        from misen.task import Task

        task_results: dict[Task, Any] = {}

        def resolve_arg(arg: Any) -> Any:
            """Resolve a task argument from cached runtime results."""
            if isinstance(arg, Task) and not arg.properties.cache:
                return task_results[arg]
            return arg

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
            )

    def as_payload(self, workspace: Workspace) -> bytes:
        """Return a serialized payload that can be executed to run the work unit."""
        return cloudpickle.dumps(functools.partial(self.execute, workspace=workspace))


def build_work_graph(tasks: set[Task], workspace: Workspace) -> DependencyGraph[WorkUnit]:
    """Given a set of tasks, transform their Task DAG into a DAG of WorkUnits."""
    # Task dependency graph: an edge from A to B means A depends on B
    union = Task((lambda *_: None), *tasks)
    task_graph: DependencyGraph[Task] = union.dependency_graph(workspace=workspace)
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
