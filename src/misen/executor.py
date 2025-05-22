from __future__ import annotations

from abc import abstractmethod
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
        return _computable_groups(task, workspace)

    @abstractmethod
    def submit(self, task: Task, workspace: Workspace) -> Future:
        raise NotImplementedError


def _computable_groups(root: Task, workspace: Workspace) -> dict[Task, list[Task]]:
    """
    This function should partition (s.t. `task` and every cache=True Task is the root of a partition).
    Then we should return the graph of partitions (i.e. a contracted graph of Tasks).
    """
    import rustworkx

    hash_to_task: dict[int, Task] = {}
    selected_nodes: set[int] = set()  # root and cacheable tasks
    dag = rustworkx.PyDiGraph()

    ### Build DAG via DFS

    hash_to_node: dict[int, int] = {}
    stack: list[Task] = [root]
    task_is_cached: dict[int, bool] = {}

    root_hash = root.__hash__()
    root_node = dag.add_node(root_hash)
    hash_to_node[root_hash] = root_node
    selected_nodes.add(root_node)

    while stack:
        task: Task = stack.pop()
        task_hash: int = task.__hash__()

        # check if already visited; else visit
        if task_hash in hash_to_task:
            continue
        hash_to_task[task_hash] = task

        # from prior iteration
        node = hash_to_node[task_hash]

        # select node if cacheable
        if task.properties.cacheable:
            selected_nodes.add(node)

        # traverse children
        for dep in task.dependencies():
            dep: Task
            dep_hash: int = dep.__hash__()

            # skip cached tasks
            if dep_hash not in task_is_cached:
                task_is_cached[dep_hash] = dep.is_cached_direct(workspace=workspace)
            if task_is_cached[dep_hash]:
                continue

            # add dependent to graph if not already present
            if dep_hash not in hash_to_node:
                hash_to_node[dep_hash] = dag.add_node(dep_hash)
                stack.append(dep)
            dep_node = hash_to_node[dep_hash]

            # add edge from task to dep_task
            dag.add_edge(node, dep_node, None)  # args: parent, child, edge_data

    ### Retain only selected tasks (and the induced graph)

    for node in set(dag.node_indices()) - selected_nodes:
        dag.remove_node_retain_edges(node)

    return dag
