from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar, Literal

import msgspec
from msgspec import Struct

from .settings import Settings
from .workspace import Workspace, WorkspaceConfig

if TYPE_CHECKING:
    from concurrent.futures import Future

    from .task import Task


class ExecutorConfig(Struct, kw_only=True):
    type: str | Literal["local"] | None = None

    def default(self) -> ExecutorConfig:
        from .executors.local import LocalExecutorConfig

        return LocalExecutorConfig(i=99)

    def from_settings(self, settings: Settings | None = None) -> ExecutorConfig:
        if settings is None:
            settings = Settings()

        if (executor_toml := settings.toml_data.get("executor", None)) is not None:
            config = msgspec.convert(executor_toml, type=ExecutorConfig)
            return msgspec.convert(executor_toml, type=config.resolve_config_type())

        return self.default()

    def resolve_executor_type(self) -> type[Executor]:
        if self.type is None:
            return self.from_settings().resolve_executor_type()

        match self.type:
            case "local":
                from .executors.local import LocalExecutor

                return LocalExecutor

        module, class_name = self.type.split(":", maxsplit=1)
        return getattr(import_module(module), class_name)

    def resolve_config_type(self) -> type[ExecutorConfig]:
        return self.resolve_executor_type().ConfigT

    def load_executor(self, settings: Settings | None = None) -> Executor:
        """Load the executor based on the configuration."""
        if self.type is None:
            config = self.from_settings(settings=settings)
        else:
            config = self
        executor_cls = config.resolve_executor_type()
        return executor_cls(config=config)


class Executor(ABC):
    ConfigT: ClassVar[type[ExecutorConfig]]

    @abstractmethod
    def __init__(self, config: ExecutorConfig):
        raise NotImplementedError

    def computable_groups(self, task: Task, workspace: Workspace | None = None):
        if workspace is None:
            workspace = WorkspaceConfig().load_workspace()
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
