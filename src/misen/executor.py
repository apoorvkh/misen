"""
Executor interface for submitting a Task's DAG to an execution backend (e.g. SLURM).

1. The Task DAG is decomposed into WorkUnits: connected subgraphs whose vertices cover the DAG.
2. WorkUnits are submitted for execution; parallel processing is subject to dependency order.
3. The yielded Jobs can be used to monitor the execution status of each submitted WorkUnit.

Pre-computed, cacheable Tasks (and descendants) are pruned ahead-of-time from the DAG.

The root and every cacheable Task in the original DAG are selected as roots for WorkUnits.
Each WorkUnit consists of the subgraph of non-cacheable Tasks (truncated at downstream,
cacheable Tasks) reachable from its root. WorkUnits are not necessarily disjoint.

WorkUnits may depend on other each other. Dependencies are executed first and their results can be
retrieved via the Workspace cache (since their roots are cacheable).
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeAlias, TypeVar, cast

from .settings import Settings
from .task import Task, TaskResources

if TYPE_CHECKING:
    from .utils.graph import DependencyGraph
    from .workspace import Workspace, WorkspaceParameters

__all__ = ["Executor"]

ExecutorType: TypeAlias = str | Literal["auto", "slurm"]
JobT = TypeVar("JobT", bound="Job")


class Executor(Generic[JobT], ABC):
    """Abstract interface for implementing an Executor for a specific backend."""

    @abstractmethod
    def _dispatch(self, function: Callable, resources: TaskResources, dependencies: set[JobT]) -> JobT:
        """For dispatching a function with the backend. Returns a Job."""

    def submit(self, task: Task, workspace: Workspace):
        """Entrypoint for submitting a Task's DAG to the Executor."""
        workspace_params = workspace.to_params()

        work_graph: DependencyGraph[WorkUnit] = _build_work_graph(root=task, workspace=workspace)
        jobs: dict[WorkUnit, JobT] = {}

        for i in work_graph.evaluation_order():
            w: WorkUnit = work_graph[i]
            jobs[w] = self._dispatch(
                functools.partial(w.execute, workspace_params=workspace_params),
                resources=w.resources,
                dependencies={jobs[d] for d in w.dependencies},
            )

    @staticmethod
    def auto(settings: Settings | None = None) -> Executor:
        if settings is None:
            settings = Settings()

        executor_type = settings.toml_data.get("executor_type", "auto")
        executor_cls = Executor._resolve_type(executor_type)
        if executor_cls is not Executor:
            return executor_cls(**settings.toml_data.get("executor_kwargs", {}))

        from misen.executors.slurm import SlurmExecutor

        return SlurmExecutor()

    @staticmethod
    def _resolve_type(t: ExecutorType) -> type[Executor]:
        match t:
            case "auto":
                return Executor
            case "slurm":
                from misen.executors.slurm import SlurmExecutor

                return SlurmExecutor
            case _:
                module, class_name = t.split(":", maxsplit=1)
                return getattr(import_module(module), class_name)


class Job(ABC):
    pass


class WorkUnit:
    """
    A unit of work for processing, corresponding to a DAG of Tasks (`self.graph`). All Tasks are non-cacheable, with the
    possible exception of the root. Tasks are executed one-by-one in dependency order. `self.resources` corresponds to
    the resources required to execute any Task in the DAG.
    """

    graph: DependencyGraph[Task]
    resources: TaskResources
    dependencies: set[WorkUnit]
    _hash: int

    def __init__(self, task: Task, dependencies: set[WorkUnit]):
        # DAG of non-cacheable Tasks (truncated at downstream, cacheable Tasks) reachable from `task`
        self.graph = task._dependency_graph(exclude_cacheable=True)

        # Union of resources for all tasks in graph
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

        # WorkUnits which must be computed before this one
        self.dependencies = dependencies

        # Tasks and WorkUnits have a 1:1 correspondence, so we can defer to Task.__hash__
        # Two Tasks may have equivalent `self.graph`s at runtime (determined after self.dependencies are resolved)
        self._hash = hash(task)

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return f"WorkUnit(hash={self._hash % 101})"

    def execute(self, workspace_params: WorkspaceParameters):
        """Execute the Tasks in self.graph one-by-one (in dependency order)."""
        workspace = workspace_params.construct()

        task_results: dict[Task, Any] = {}

        evaluation_order = [self.graph[t] for t in self.graph.evaluation_order()]
        for i, task in enumerate(evaluation_order):
            task_results[task] = Task(
                task.func,
                *tuple(task_results[v] if isinstance(v, Task) and v in task_results else v for v in task.args),
                **{
                    k: task_results[v] if isinstance(v, Task) and v in task_results else v
                    for k, v in task.kwargs.items()
                },
            ).result(workspace=workspace, compute_if_uncached=True)

            # remove any results that are not needed by future dependents
            remaining_tasks = evaluation_order[i + 1 :]
            if len(remaining_tasks) > 0:
                remaining_deps = set.union(*(t._dependencies for t in remaining_tasks))
            else:
                remaining_deps = set()

            cached_tasks = set(task_results.keys())
            for t in cached_tasks - remaining_deps:
                del task_results[t]


def _build_work_graph(root: Task, workspace: Workspace) -> DependencyGraph[WorkUnit]:
    """
    Given `root: Task`, transform its DAG of Tasks (excluding already cached subgraphs) into a DAG of WorkUnits.
    """
    # dependency graph: dependents point to dependencies
    task_graph: DependencyGraph[Task] = root._dependency_graph(exclude_cached=True, workspace=workspace)
    # Retain only root & cachable tasks (and the induced graph minor)
    anchor_graph = task_graph.copy()
    anchor_graph.coarsen_to_anchors(
        anchors=[i for i in anchor_graph.node_indices() if anchor_graph[i] == root or anchor_graph[i].properties.cache]
    )

    # replace nodes with WorkUnit instances
    work_graph = cast("DependencyGraph[WorkUnit]", anchor_graph.copy())
    for i in work_graph.evaluation_order():
        work_graph[i] = WorkUnit(task=anchor_graph[i], dependencies=set(work_graph.successors(i)))

    return work_graph
