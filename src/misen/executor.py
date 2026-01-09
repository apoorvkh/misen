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
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeAlias, TypeVar, cast, get_args

from typing_extensions import assert_never

from misen.utils.hashes import short_hash
from misen.utils.settings import FromSettingsABC

from .task import Task, TaskResources

if TYPE_CHECKING:
    from .utils.graph import DependencyGraph
    from .workspace import Workspace

__all__ = ["Executor"]

ExecutorType: TypeAlias = Literal["local", "slurm"]


class Job(ABC):
    @abstractmethod
    def state(self) -> Literal["pending", "running", "done", "failed", "unknown"]: ...


JobT = TypeVar("JobT", bound=Job)

# TODO: submit array?


class Executor(Generic[JobT], FromSettingsABC):
    """Abstract interface for implementing an Executor for a specific backend."""

    @abstractmethod
    def _dispatch(self, function: Callable, resources: TaskResources, dependencies: set[JobT]) -> JobT:
        """For dispatching a function with the backend. Returns a Job."""

    def submit(self, task: Task, workspace: Workspace):
        """Entrypoint for submitting a Task's DAG to the Executor."""
        work_graph: DependencyGraph[WorkUnit] = _build_work_graph(root=task, workspace=workspace)
        jobs: dict[WorkUnit, JobT] = {}

        for i in work_graph.evaluation_order():
            w: WorkUnit = work_graph[i]
            if w.root.is_cached(workspace=workspace):
                continue
            jobs[w] = self._dispatch(
                functools.partial(w.execute, workspace=workspace),
                resources=w.resources,
                dependencies={jobs[d] for d in w.dependencies if d in jobs},
            )

        return jobs

    @staticmethod
    def _settings_key() -> str:
        return "executor"

    @staticmethod
    def _default() -> Executor:
        from misen.executors.slurm import SlurmExecutor

        return SlurmExecutor()

    @classmethod
    def _resolve_type(cls, type_name: str | ExecutorType) -> type["Executor"]:
        if type_name in get_args(ExecutorType):
            type_name = cast("ExecutorType", type_name)
            match type_name:
                case "local":
                    from .executors.local import LocalExecutor

                    return LocalExecutor
                case "slurm":
                    from .executors.slurm import SlurmExecutor

                    return SlurmExecutor
                case _:
                    assert_never(type_name)
        return super()._resolve_type(type_name)


class WorkUnit:
    """
    A unit of work for processing, corresponding to a DAG of Tasks (`self.graph`). All Tasks are non-cacheable, with the
    possible exception of the root. Tasks are executed one-by-one in dependency order. `self.resources` corresponds to
    the resources required to execute any Task in the DAG.
    """

    root: Task
    graph: DependencyGraph[Task]
    resources: TaskResources
    dependencies: set[WorkUnit]

    def __init__(self, root: Task, dependencies: set[WorkUnit]):
        self.root = root
        # WorkUnits which must be computed before this one
        self.dependencies = dependencies

        # DAG of non-cacheable Tasks (truncated at downstream, cacheable Tasks) reachable from `task`
        self.graph = root._dependency_graph(exclude_cacheable=True)

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

    def __hash__(self):
        # Tasks and WorkUnits have a 1:1 correspondence, so we can defer to Task.__hash__
        # Two Tasks may have equivalent `self.graph`s at runtime (determined after self.dependencies are resolved)
        return hash(self.root)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WorkUnit) and self.root == other.root

    def __repr__(self):
        return f"WorkUnit(hash={short_hash(self)})"

    def execute(self, workspace: Workspace):
        """Execute the Tasks in self.graph one-by-one (in dependency order)."""
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
    task_graph: DependencyGraph[Task] = root._dependency_graph(workspace=workspace)
    # Retain only root & cachable tasks (and the induced graph minor)
    anchor_graph = task_graph.copy()
    anchor_graph.coarsen_to_anchors(
        anchors=[i for i in anchor_graph.node_indices() if anchor_graph[i] == root or anchor_graph[i].properties.cache]
    )

    # replace nodes with WorkUnit instances
    work_graph = cast("DependencyGraph[WorkUnit]", anchor_graph.copy())
    for i in work_graph.evaluation_order():
        work_graph[i] = WorkUnit(root=anchor_graph[i], dependencies=set(work_graph.successors(i)))

    return work_graph
