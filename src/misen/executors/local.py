from __future__ import annotations

from typing import TYPE_CHECKING

from ..executor import Executor

if TYPE_CHECKING:
    from ..task import Task
    from ..workspace import Workspace


class LocalExecutor(Executor):
    def __init__(self) -> None:
        pass

    # TODO: figure out a unified interface
    # Should return Future to each task

    def _submit(self, dependency_graph: dict[Task, set[Task]], workspace: Workspace) -> None:
        unmet_deps: dict[Task, set[Task]] = {t: d.copy() for t, d in dependency_graph.items()}

        while len(unmet_deps) > 0:
            task = next(k for k, v in unmet_deps.items() if len(v) == 0)
            del unmet_deps[task]
            for t in unmet_deps.keys():
                unmet_deps[t].discard(task)

            # TODO: this should be async
            task.result(
                workspace=workspace,
                compute_if_uncached=True,
                compute_uncached_deps=True,
            )


# from typing import Any
# import asyncio
# from rustworkx import topological_sort

# class MultithreadedLocalExecutor(Executor):
#     def submit(self, task: Task, workspace: Workspace):
#         def execute_task_local(t: Task):
#             return t.result(workspace=workspace)

#         async def async_helper() -> Any:
#             task_graph = self.computable_groups(task, workspace=workspace)

#             # graphviz_draw(task_graph, edge_attr_fn=label_dag_edge).save("dag.png")

#             task_dict: dict[int, asyncio.Awaitable] = {}  # is this the right typing? # type: ignore

#             # go through tasks in topological order
#             for node in topological_sort(task_graph):
#                 # get inbound edges to this task
#                 in_edges = task_graph.in_edges(node)
#                 # if there are inbound edges ("dependencies")
#                 if in_edges != [] or len(task_dict) != 0:
#                     # check whether we need to wait on anything
#                     # construct temporary task list to use asyncio.gather on...
#                     tasks = []
#                     tasks_keys = []
#                     for in_edge in in_edges:
#                         # the following node is a dependency of this node...
#                         dependency = in_edge[0]
#                         # if this dependency is not yet gathered, it'll be in here
#                         if dependency in task_dict:
#                             print(f"{node} waiting for {dependency}")
#                             tasks.append(task_dict[dependency])
#                             tasks_keys.append(dependency)
#                     # gather dependencies
#                     await asyncio.gather(*tasks)
#                     # delete dependencies (they're now done) so no other
#                     # tasks ever need to wait on them
#                     for k in tasks_keys:
#                         del task_dict[k]

#                 # run this task
#                 print(f"running {node}")
#                 new_task = asyncio.create_task(
#                     asyncio.to_thread(execute_task_local, task_graph[node])
#                 )
#                 task_dict[node] = new_task

#             return await new_task  # type: ignore

#         return async_helper
