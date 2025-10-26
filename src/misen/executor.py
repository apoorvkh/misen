from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, Literal

from .settings import Settings
from .workspace import Workspace

if TYPE_CHECKING:
    from concurrent.futures import Future

    from .task import Task

__all__ = ["Executor"]

ExecutorType = str | Literal["auto", "local", "slurm"]


class Executor(ABC):
    @staticmethod
    def auto(settings: Settings | None = None) -> "Executor":
        if settings is None:
            settings = Settings()

        executor_type = settings.toml_data.get("executor_type", "auto")
        executor_cls = Executor._resolve_type(executor_type)
        if executor_cls is not Executor:
            return executor_cls(**settings.toml_data.get("executor_kwargs", {}))

        # default
        from misen.executors.local import LocalExecutor

        return LocalExecutor(i=20)

    @staticmethod
    def _resolve_type(t: ExecutorType) -> type["Executor"]:
        match t:
            case "auto":
                return Executor
            case "local":
                from misen.executors.local import LocalExecutor

                return LocalExecutor
            case "slurm":
                raise NotImplementedError
            case _:
                module, class_name = t.split(":", maxsplit=1)
                return getattr(import_module(module), class_name)

    @abstractmethod
    def submit(self, task: Task, workspace: Workspace) -> Future: ...

    def _computable_groups(self, task: Task, workspace: Workspace | None = None):
        if workspace is None:
            workspace = Workspace.auto()
        return distributable_tasks(task, workspace)


def distributable_tasks(root: Task, workspace: Workspace) -> dict[Task, list[Task]]:
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

    nodes_to_remove = dag.filter_nodes(lambda task: not (task.properties.cacheable or task == root))
    for node in nodes_to_remove:
        dag.remove_node_retain_edges(node)

    ### Return as adjacency list

    return {dag.get_node_data(node): dag.successors(node) for node in dag.node_indices()}
