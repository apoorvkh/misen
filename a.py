import numpy as np

from misen import Experiment, task, Task
from misen.utils.task_graph_builder import TaskGraphBuilder
from misen.utils.det_hash import deterministic_hashing


@task(uuid="QuNP")
def add(x, y):
    return x + y


@task(uuid="def")
def multiply(x, y, z: int = 0, **kwargsx):
    return x * y


def double(x):
    return x * 2


# class MultiplyExperiment(Experiment):
#     def calls(self):
#         return multiply(add(double(1), 4), add(np.csingle(3), 4), hello=np.datetime64("2005-02-25"))


if __name__ == "__main__":
    with TaskGraphBuilder(globals()):
        task_graph: Task = multiply(
            add(double(1), 4), add(np.csingle(3), 4), hello=np.datetime64("2005-02-25")
        )

    print(task_graph)

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
