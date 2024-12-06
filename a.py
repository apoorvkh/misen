import matplotlib.pyplot as plt
import numpy as np
from rustworkx import topological_sort
from rustworkx.visualization import mpl_draw, graphviz_draw

from misen import Executor, Task, TaskGraphBuilder, task
from misen.executor import ExecNode
from misen.workspace import TestWorkSpace


@task(uuid="QuNP", cache=False)
def add(addx, addy):
    return addx + addy

@task(uuid="add2", cache=True)
def adds(addsx, addsy):
    return addsx + addsy

@task(uuid="def", cache=True)
def multiply(mulx, muly, mulz: int = 0, **mulkwargsx):
    return mulx * muly


def double(dubx):
    return dubx * 2


# class MultiplyExperiment(Experiment):
#     def calls(self):
#         return multiply(add(double(1), 4), add(np.csingle(3), 4), hello=np.datetime64("2005-02-25"))


if __name__ == "__main__":
    with TaskGraphBuilder(globals()):
        a = adds(4,3)
        b = adds(2,1)
        task_graph: Task = multiply(
            multiply(
                add(
                    add(a, b),
                    double(1)), 
                b),
            add(
                add(
                    np.csingle(3), 
                    a), 
                b),
            hello=np.datetime64("2005-02-25"),
        )

    e = Executor()
    
    dag = e.computable_groups(task=task_graph, workspace=TestWorkSpace())

    def label_dag_node(n):
        return {'label': str(n)}
    
    def label_dag_edge(e):
        return {'label': str(e)}
    
    def label_partition_node(n):
        return {'label': str((n.func.__name__, n.hash))}

    graphviz_draw(dag, node_attr_fn=label_dag_node, edge_attr_fn=label_dag_edge).save("dag.png")
    #graphviz_draw(partitions, node_attr_fn=label_partition_node, edge_attr_fn=label_dag_edge).save("partition_dag.png")

    #print(partitions.nodes())
    #print(partitions.edges())
    #print(partitions.edge_index_map())

# task_graph: Task = multiply(
#     add(x=double(1), y=4), add(x=np.csingle(3), y=4), hello=np.datetime64("2005-02-25")
# )
# Task(
#     func=__main__.multiply,
#     kwargs={
#         "x": Task(func=__main__.add, kwargs={"x": 2, "y": 4}),
#         "y": Task(func=__main__.add, kwargs={"x": (3 + 0j), "y": 4}),
#         "kwargsx": {"hello": numpy.datetime64("2005-02-25")},
#         "z": 0,
#     },
# )
