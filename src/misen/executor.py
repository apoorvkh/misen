from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Callable
from rustworkx import PyDAG, topological_sort
from dataclasses import dataclass

from misen.task import Task
from misen.workspace import Workspace

if TYPE_CHECKING:
    from .task import Task
    from .workspace import Workspace

# cacheable=True, necessarily
@dataclass
class PartitionNode:
    func: Callable
    hash: int

@dataclass
class ExecNode:
    func: Callable
    cacheable: bool
    hash: int


class Executor(ABC):
    def computable_groups(self, task: Task, workspace: Workspace):
        """
        This function should partition (s.t. `task` and every cache=True Task is the root of a partition).
        Then we should return the graph of partitions. Each partition should represented as a topologically-sorted list of its Tasks.

        I think we can use the rustworkx library if want to use efficient graph algorithms.
        """
        # TODO: implement this here

        # this section builds `dag`: just reformating the task graph as a rustworkx PyDAG
        dag = PyDAG()

        root = ExecNode(func=task.func, cacheable=task.properties.cacheable, hash=task.__hash__())
        root_node = dag.add_node(root)

        # more or less a 'seen' dictionary, for making sure that a single node can connect to multiple 'parents'
        d = {}

        def rec(cur: Task | Any, parent: int, kwarg: str):

            # at the moment, I haven't turned constants into their own kind of Task
            if isinstance(cur, Task):
                # check if this node seen alr
                if cur.__hash__() in d:
                    n = d[cur.__hash__()]
                else:
                    # add it to the graph
                    n = dag.add_node(ExecNode(func=cur.func, cacheable=cur.properties.cacheable, hash=cur.__hash__()))
                    if cur.properties.cacheable:
                        d[cur.__hash__()] = n
                    # recurse on children
                    for k, child in cur.kwargs.items():
                        rec(child, n, k)
                # edges contain the kwarg names...
                dag.add_edge(n, parent, kwarg)
            else:
                # this is a leaf node (indegree 0)
                n = dag.add_node(cur)
                dag.add_edge(n, parent, kwarg)
        
        # construct dag
        for k, child in task.kwargs.items():
            rec(child, root_node, k)

        # now, must partition graph into partitions: all cacheable nodes of dag are nodes
        # of the following graph. Edges still correspond to data dependencies (in the form of kwargs)
        # but since partitions represent subgraphs, we also include the hash of the task that the kwarg
        # is destined for. It's weird. Look at the image examples.
        partitions = PyDAG()

        root_partition = PartitionNode(func=root.func, hash=root.hash)
        root_partition_node = partitions.add_node(root_partition)

        dag_nodes = dag.nodes()

        d = {}

        def rec_partition(cur_node: int, parent_partition: int, parent_node_hash: int, edge_kwarg: str | None):

            cur = dag_nodes[cur_node]
            # again... nodes not all ExecNodes
            if isinstance(cur, ExecNode):
                exists = cur.hash in d
                n = parent_partition
                # we need to make a new PartitionNode
                if cur.cacheable and not exists:
                    n = partitions.add_node(PartitionNode(func=cur.func, hash=cur.hash))
                    # simply keep track of the node this hash corresponds with, so we 
                    # can reuse it if we come accross the same node again
                    d[cur.hash] = n
                    partitions.add_edge(n, parent_partition, (parent_node_hash, edge_kwarg))
                elif exists:
                    # otherwise, we link the existing partition to the parent partition
                    partitions.add_edge(d[cur.hash], parent_partition, (parent_node_hash, edge_kwarg))
                # necessary, if not cacheable
                if not exists:
                    # not cacheable so we never make a new partition, so doesn't need to be in d
                    # d[cur.hash] = n
                    # need to recurse on children, since their subgraphs will potentially create
                    # new partitions.
                    for child, kwarg in dag.adj_direction(cur_node, True).items():
                        rec_partition(child, n, cur.hash, kwarg)
            # non-ExecNode nodes never appear in the partition graph since they're not cacheable
                    
        
        for child, kwarg in dag.adj_direction(root_node, True).items():
            rec_partition(child, root_partition_node, root.hash, kwarg)

        return dag, partitions

        # TODO: remove this old impl

        # partitions = PyDAG()

        # root_partition = Partition(PyDAG())
        # root = partitions.add_node(root_partition)

        # def construct_partition_graph(current_task: Task | Any, current_partition: Partition, parent: int | None, parent_partition: int, kwarg: Any | None = None):
            
        #     if not isinstance(current_task, Task):
        #         # add to current_partition as node, with edge going to parent
        #         n = current_partition.tasks.add_node(current_task)
        #         # add and label edge with kwarg of this task in parent task
        #         current_partition.tasks.add_edge(n, parent, kwarg)
        #         return

        #     # ignore cached tasks
        #     #if current_task.is_cached(workspace=workspace):
        #     #    return

        #     print(current_task, current_task.properties.cacheable)
            
        #     if not current_task.properties.cacheable:

        #         # add to current_partition as node, with edge going to parent
        #         n = current_partition.tasks.add_node(current_task)
        #         # add and label edge with kwarg of this task in parent task
        #         current_partition.tasks.add_edge(n, parent, kwarg)
        #         # for kwargs of this task, recurse
        #         for k, task in current_task.kwargs.items():
        #             # same partition and partition parent, we're the parent (in the partition), with kwarg k
        #             construct_partition_graph(task, current_partition, n, parent_partition, k)
        #     else:
                
        #         # partition = current_partition
        #         # partition_node = partition_parent
        #         # partition_root = partition_parent
        #         # we have a new partition
        #         # if not :
        #         partition = Partition(tasks=PyDAG())
        #         # and must add it to the graph, with an edge towards the parent. We can once again use the kwarg to label the edge
        #         partition_node = partitions.add_node(partition)
        #         if parent_partition is not None:
        #             partitions.add_edge(partition_node, parent_partition, kwarg)
        #         # this task is the root of this partition
        #         partition_root = partition.tasks.add_node(current_task)
        #         # else:
        #         #     n = partition.tasks.add_node(current_task)
        #         #     partition.tasks.add_edge(n, parent, None)
        #         #     partition_root = n
        #         # recurse on children
        #         for k, task in current_task.kwargs.items():
        #             # new partition, we're the parent (in the partition), with kwarg k
        #             construct_partition_graph(task, partition, partition_root, partition_node, k)

        # construct_partition_graph(task, root_partition, None, root, None)

        # return partitions

    def submit(self, task: Task, workspace: Workspace):
        raise NotImplementedError


# TODO: implement LocalExecutor that implements local / async multi-processing / multi-threading

# TODO: implement SlurmExecutor based on submitit
