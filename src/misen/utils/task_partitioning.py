from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import rustworkx
from rustworkx import PyDAG, topological_sort
from rustworkx.visualization import graphviz_draw
from rustworkx.visit import DFSVisitor, PruneSearch

from ..task import Task

if TYPE_CHECKING:
    from ..workspace import Workspace


def computable_groups(task: Task, workspace: Workspace):
    """
    This function should partition (s.t. `task` and every cache=True Task is the root of a partition).
    Then we should return the graph of partitions. Each partition should represented as a topologically-sorted list of its Tasks.
    """
    # TODO: implement this here

    # this section builds `dag`: just reformating the task graph as a rustworkx PyDAG
    dag = PyDAG()

    root = task
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
                n = dag.add_node(cur)
                if cur.properties.cacheable:
                    d[cur.__hash__()] = n
                # recurse on children
                for i, child in enumerate(cur.args):
                    rec(child, n, f"_args_{i}")
                for k, child in cur.kwargs.items():
                    rec(child, n, k)
            # edges contain the kwarg names...
            dag.add_edge(n, parent, kwarg)
        else:
            # this is a leaf node (indegree 0)
            n = dag.add_node(cur)
            dag.add_edge(n, parent, kwarg)

    # Step 1: Construct DAG
    for i, child in enumerate(task.args):
        rec(child, root_node, f"_args_{i}")
    for k, child in task.kwargs.items():
        rec(child, root_node, k)

    # Step 2: Copy DAG and snip outgoing edges from cacheable nodes,
    #  compute connected components

    # graphviz_draw(dag).save("dag.png")

    # first, make note of the index of each node within the each node's data
    for node in dag.node_indices():
        dag[node] = (dag[node], node)

    dag_c = copy.copy(dag)
    for src, dest in dag_c.edge_list():
        t: Task | Any = dag_c.get_node_data(src)[0]
        if isinstance(t, Task) and t.properties.cacheable:
            dag_c.remove_edge(src, dest)
    # graphviz_draw(dag, edge_attr_fn=label_dag_edge).save("dag_.png")
    # graphviz_draw(dag_c, edge_attr_fn=label_dag_edge).save("dag__.png")
    conn_comps = rustworkx.connected_components(dag_c.to_undirected())  # type: ignore
    # print(conn_comps)
    # Step 3: Contract connected components to their topological root node

    new_root = 0
    i = 0
    for conn_comp in conn_comps:
        conn_comp_lst = list(reversed(list(conn_comp)))
        sub_dag = dag.subgraph(conn_comp_lst)
        sort = topological_sort(sub_dag)
        conn_comp_root = sub_dag[sort[-1]][1]
        new_node = dag.contract_nodes(conn_comp_lst, dag[conn_comp_root][0])
        if conn_comp_root == 0:
            new_root = new_node
        # graphviz_draw(dag, edge_attr_fn=label_dag_edge).save(f"dag__{i}.png")
        i += 1

    # Step 4: Traverse and delete nodes that are already cached
    # and get the connected component containing the root node
    class CachedVisitor(DFSVisitor):
        def __init__(self):
            self.to_remove = []

        def tree_edge(self, edge):  # type: ignore
            t: Task | Any = dag[edge[0]]
            if isinstance(t, Task) and t.is_cached(workspace=workspace):
                self.to_remove.append(edge[0])
                raise PruneSearch()

    # find nodes to remove...
    vis = CachedVisitor()
    rustworkx.dfs_search(dag, [new_root], vis)

    # remove nodes
    for node in vis.to_remove:
        dag.remove_node(node)

    computable_subdag = dag.subgraph(list(rustworkx.ancestors(dag, new_root)) + [new_root])
    # graphviz_draw(dag, edge_attr_fn=label_dag_edge).save("dag_final.png")
    # print(new_root, dag.node_indices(), list(rustworkx.ancestors(dag, new_root)))
    return computable_subdag
