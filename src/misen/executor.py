from __future__ import annotations

from abc import abstractmethod
from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, Literal

import msgspec
from msgspec import Struct

from .settings import Settings
from .workspace import Workspace

if TYPE_CHECKING:
    from concurrent.futures import Future

    from .task import Task


class Executor(Struct, kw_only=True):
    type: str | Literal["local"] | None = None

    @staticmethod
    def load(settings: Settings | None = None) -> Executor:
        settings = settings or Settings()

        if "executor" in settings.toml_data:
            executor = msgspec.convert(settings.toml_data["executor"], type=Executor)
            executor_cls: type[Executor] | None = executor._resolve_type()
            if executor_cls is not None:
                return msgspec.convert(
                    settings.toml_data["executor"],
                    type=executor_cls,
                )

        # fallback to default
        from .executors.local import LocalExecutor

        return LocalExecutor(i=99)

    def _resolve_type(self) -> type[Executor] | None:
        if self.type is None:
            return None

        match self.type:
            case "local":
                from .executors.local import LocalExecutor

                return LocalExecutor

        module, class_name = self.type.split(":", maxsplit=1)
        return getattr(import_module(module), class_name)

    def computable_groups(self, task: Task, workspace: Workspace | None = None):
        workspace = workspace or Workspace.load()
        return _distributable_tasks(task, workspace)

    @abstractmethod
    def submit(self, task: Task, workspace: Workspace) -> Future:
        raise NotImplementedError


def _distributable_tasks(root: Task, workspace: Workspace) -> dict[Task, list[Task]]:
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
        for dep in task.dependencies():
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
