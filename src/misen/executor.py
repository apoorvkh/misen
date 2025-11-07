from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, Callable, Generic, Literal, TypeAlias, TypeVar

from .settings import Settings

if TYPE_CHECKING:
    from .task import Task, TaskResources
    from .workspace import Workspace

__all__ = ["Executor"]


T = TypeVar("T")
AdjacencyList: TypeAlias = dict[T, set[T]]


class Job(ABC):
    pass


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

        # default
        # from misen.executors.local import LocalExecutor

        # return LocalExecutor()
        from misen.executors.slurm import SlurmExecutor

        return SlurmExecutor()

    @staticmethod
    def _resolve_type(t: str | Literal["auto", "local", "slurm"]) -> type[Executor]:
        match t:
            case "auto":
                return Executor
            case "local":
                raise NotImplementedError("LocalExecutor is not implemented yet")
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

    def submit(self, task: Task, workspace: Workspace) -> AdjacencyList[JobT]:
        task_graph: AdjacencyList[Task] = distributable_tasks(task, workspace)

        remaining_deps: AdjacencyList[Task] = {t: d.copy() for t, d in task_graph.items()}
        jobs: dict[Task, JobT] = {}

        while len(remaining_deps) > 0:
            task = next(k for k, v in remaining_deps.items() if len(v) == 0)
            del remaining_deps[task]
            for t in remaining_deps.keys():
                remaining_deps[t].discard(task)

            # TODO: new implementation for task.result with multiprocessing

            execution_fn = functools.partial(
                task.result,
                workspace=workspace,
                compute_if_uncached=True,
                compute_uncached_deps=True,
            )

            # TODO: use union of task resources

            resources = task.resources

            dependencies = {jobs[d] for d in task_graph[task]}

            jobs[task] = self._submit(
                execution_fn,
                resources=resources,
                dependencies=dependencies,
            )

        job_graph: AdjacencyList[JobT] = {jobs[t]: {jobs[d] for d in task_graph[t]} for t in jobs}

        return job_graph


def distributable_tasks(root: Task, workspace: Workspace) -> AdjacencyList[Task]:
    """
    Given a DAG (represented by the root Task), this function should identify tasks that can be executed by distributed workers.
    Specifically, these are the cacheable tasks. This function should return a dependency graph of just those tasks and the root.

    Implementation: build a DAG, then remove all (non-cacheable or non-root) nodes, while maintaining the dependency structure.
    Returns: DAG in adjacency list format. Keys depend on values.
    """
    from rustworkx import PyDiGraph

    dag = PyDiGraph()

    ### Build DAG via DFS

    @cache
    def _is_cached(task: Task) -> bool:
        return task.is_cached(workspace=workspace)

    task_node: dict[Task, int] = {root: dag.add_node(root)}
    stack: list[Task] = [root]
    visited: set[Task] = set()

    while stack:
        task: Task = stack.pop()
        if task in visited:
            continue
        visited.add(task)

        # from prior iteration
        node = task_node[task]

        # traverse children
        for dep in task._dependencies:
            dep: Task
            # skip cached tasks
            if _is_cached(dep):
                continue

            # add dependent to graph if not already present
            if dep not in task_node:
                task_node[dep] = dag.add_node(dep)
                stack.append(dep)
            dep_node = task_node[dep]

            # add edge from task to dep_task
            dag.add_edge(node, dep_node, None)  # args: parent, child, edge_data

    ### Retain only root & cachable tasks (and the induced graph minor)

    nodes_to_remove = dag.filter_nodes(lambda task: not (task.properties.cache or task == root))
    for node in nodes_to_remove:
        dag.remove_node_retain_edges(node)

    ### Return as adjacency list

    return {dag.get_node_data(node): set(dag.successors(node)) for node in dag.node_indices()}
