import asyncio
import time

import numpy as np
from rustworkx.visualization import graphviz_draw

from misen import LocalExecutor, Task, TaskGraphBuilder, task
from misen.workspace import TestWorkSpace


@task(uuid="QuNP", cache=False)
def add(addx, addy):
    return addx + addy

@task(uuid="add2", cache=True)
def adds(addsx, addsy):
    return addsx + addsy

@task(uuid="def", cache=True)
def multiply(mulx, muly, mulz: int = 0, **mulkwargsx):
    print("multiplying for 1 second")
    time.sleep(1)
    return mulx * muly

@task(uuid="delayret", cache=True)
def ret(a, t):
    print(f"sleeping for {t}")
    time.sleep(t)
    return a


def double(dubx):
    return dubx * 2


# class MultiplyExperiment(Experiment):
#     def calls(self):
#         return multiply(add(double(1), 4), add(np.csingle(3), 4), hello=np.datetime64("2005-02-25"))


if __name__ == "__main__":
    with TaskGraphBuilder(globals()):
        # a = adds(4,3)
        # b = adds(2,1)
        # task_graph: Task = multiply(
        #     multiply(
        #         add(
        #             add(a, b),
        #             double(1)), 
        #         b),
        #     add(
        #         add(
        #             np.csingle(3), 
        #             a), 
        #         b),
        #     hello=np.datetime64("2005-02-25"),
        # )
        task_graph: Task = multiply(
            ret(5, 3),
            ret(6, 1)
            )

    e = LocalExecutor()
    ws = TestWorkSpace()

    f = task_graph.run(workspace=ws, executor=e)

    asyncio.run(f()) # type: ignore