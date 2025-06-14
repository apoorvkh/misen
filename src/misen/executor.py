from __future__ import annotations

from abc import abstractmethod
from functools import cache
from typing import TYPE_CHECKING, Literal

from .settings import ConfigABC, TargetABC
from .workspace import Workspace, WorkspaceConfig

if TYPE_CHECKING:
    from concurrent.futures import Future

    from .task import Task


class ExecutorConfig(ConfigABC["ExecutorConfig", "Executor"], kw_only=True):
    type: str | Literal["local"] | None = None

    @staticmethod
    def settings_key() -> str:
        return "executor"

    def default(self) -> ExecutorConfig:
        from .executors.local import LocalExecutorConfig

        return LocalExecutorConfig(i=99)

    def resolve_target_type(self) -> type[Executor]:
        match self.type:
            case "local":
                from .executors.local import LocalExecutor

                return LocalExecutor
        return super().resolve_target_type()


class Executor(TargetABC[ExecutorConfig]):
    def computable_groups(self, task: Task, workspace: Workspace | None = None):
        if workspace is None:
            workspace = WorkspaceConfig().load_target()
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
        return workspace.is_cached(task=task)

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
