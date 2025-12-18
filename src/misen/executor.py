"""
This module defines the Executor interface for submitting a Task's DAG to an execution backend
(e.g. SLURM).

1. The Task DAG is decomposed into WorkUnits rooted at "anchor" Tasks.
2. Each WorkUnit is submitted to the backend. WorkUnits may be executed in parallel, subject to
   dependency order between them.
3. The Executor yields Jobs, which can be used to monitor the execution status of each submitted
   WorkUnit.

Cacheable Tasks that are already computed are pruned from the DAG (along with their descendants).
Anchors are then chosen at cache boundaries: the root Task and every cacheable Task is an anchor.
Each WorkUnit consists of an anchor and the reachable subgraph (of non-cacheable Tasks; truncated
at downstream, cacheable anchors). WorkUnits form a cover of the DAG and may overlap, but are not
necessarily disjoint. A WorkUnit may depend on other WorkUnits (and their anchors). Dependencies
are executed first, and their anchor results can be retrieved via the Workspace cache.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeAlias, TypeVar, cast

import rustworkx as rx

from .settings import Settings
from .task import Task, TaskResources

if TYPE_CHECKING:
    from .task import AdjacencyList
    from .workspace import Workspace, WorkspaceParameters

__all__ = ["Executor"]

ExecutorType: TypeAlias = str | Literal["auto", "slurm"]
JobT = TypeVar("JobT", bound="Job")


class Executor(Generic[JobT], ABC):
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

    @abstractmethod
    def _submit(
        self, function: Callable, resources: TaskResources, dependencies: set[JobT]
    ) -> JobT: ...

    def submit(self, task: Task, workspace: Workspace):  # -> AdjacencyList[JobT]:
        workspace_params = workspace.to_params()

        submission_graph, submissions = _build_submission_graph(root=task, workspace=workspace)

        jobs: dict[WorkUnit, JobT] = {}

        for work in submissions:
            jobs[work] = self._submit(
                functools.partial(work.execute, workspace_params=workspace_params),
                resources=work.resources,
                dependencies={jobs[d] for d in work.dependencies},
            )


class WorkUnit:
    graph: AdjacencyList[Task]
    resources: TaskResources
    dependencies: set[WorkUnit]
    _task_hash: int

    def __init__(self, task: Task, dependencies: set[WorkUnit]):
        self.graph = task._dependency_tree(exclude_cacheable=True)

        _resource_list: list[TaskResources] = [task.resources for task in self.graph]
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

        self.dependencies = dependencies
        self._task_hash = hash(task)

    def __hash__(self):
        return self._task_hash

    def execute(self, workspace_params: WorkspaceParameters):
        workspace = workspace_params.construct()

        task_graph: rx.PyDiGraph = _adj_list_to_rustworkx(self.graph)
        task_results: dict[Task, Any] = {}

        dependency_order = rx.topological_sort(task_graph)
        for task_index in dependency_order:
            task: Task = task_graph[task_index]
            task_results[task] = Task(
                task.func,
                *tuple(
                    task_results[v] if isinstance(v, Task) and v in task_results else v
                    for v in task.args
                ),
                **{
                    k: task_results[v] if isinstance(v, Task) and v in task_results else v
                    for k, v in task.kwargs.items()
                },
            ).result(workspace=workspace)

        # TODO: prune results if no longer needed by future dependents


class Job(ABC):
    pass


def _build_submission_graph(
    root: Task, workspace: Workspace
) -> tuple[AdjacencyList[WorkUnit], list[WorkUnit]]:
    # """
    # Given a DAG, this function should identify tasks that can be executed by distributed workers.
    # Specifically, these are the (1) root task and (2) cacheable tasks. Should return a dependency graph of just those tasks.

    # Implementation: build a DAG, then remove all (non-cacheable or non-root) nodes, while maintaining the dependency structure.
    # Returns: DAG in adjacency list format. Keys depend on values.
    # """
    ### dependency graph: dependents point to dependencies

    graph: AdjacencyList[Task] = root._dependency_tree(exclude_cached=True, workspace=workspace)
    graph: rx.PyDiGraph = _adj_list_to_rustworkx(graph)

    ### Retain only root & cachable tasks (and the induced graph minor)

    nodes_to_remove = graph.filter_nodes(lambda task: not (task.properties.cache or task == root))
    for node in nodes_to_remove:
        graph.remove_node_retain_edges(node)  # TODO: inefficient

    ###

    order = rx.topological_sort(graph)[::-1]  # dependencies first

    # replace nodes with WorkUnit instances
    for node in order:
        task: Task = graph[node]
        dependencies: set[WorkUnit] = {graph[d] for d in graph.successors(node)}
        graph[node] = WorkUnit(task=task, dependencies=dependencies)

    dependency_graph: AdjacencyList[WorkUnit] = {
        graph[node]: set(graph.successors(node)) for node in graph.node_indices()
    }

    order = [graph[node] for node in order]

    return dependency_graph, order


def _adj_list_to_rustworkx(graph: AdjacencyList[Task]) -> rx.PyDiGraph:
    dag = rx.PyDiGraph(check_cycle=True, multigraph=False)
    nodes: dict[Task, int] = {t: dag.add_node(t) for t in graph}
    for t in graph:
        n = nodes[t]
        for d in graph[t]:
            dag.add_edge(n, nodes[d], None)
    return dag
