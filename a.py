# import asyncio
import time

from misen import Executor, Experiment, Task, Workspace, task


@task(uuid="QuNP", cache=False)
def add(addx, addy):
    return addx + addy


@task(uuid="add2", cache=True)
def adds(addsx, addsy):
    return addsx + addsy


@task(uuid="def", cache=True)
def multiply(mulx, muly, mulz: int = 0, n="m", **mulkwargsx):
    print(f"{n} multiplying for 1 second")
    time.sleep(1)
    print(f"{n} returning")
    return mulx * muly


@task(uuid="delayret", cache=True)
def ret(a, t, n="ret"):
    print(f"{n} sleeping for {t} seconds")
    time.sleep(t)
    print(f"{n} returning")
    return a


def double(dubx):
    return dubx * 2


# class MultiplyExperiment(Experiment):
#     def calls(self):
#         return multiply(add(double(1), 4), add(np.csingle(3), 4), hello=np.datetime64("2005-02-25"))


if __name__ == "__main__":
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
    task_graph = Task(
        multiply,
        Task(multiply, Task(ret, 2, 3, n="r1"), 4, n="m1"),
        Task(ret, 6, 1, n="r2"),
        n="m2",
    )

    class TestExperiment(Experiment):
        def tasks(self) -> dict[str, Task]:
            a = Task(multiply, Task(ret, 2, 3, n="r1"), 4, n="m1")

            return {
                "subresult": a,
                "multresult": Task(
                    multiply,
                    a,
                    Task(ret, 6, 1, n="r2"),
                    n="m2",
                ),
            }

    class TestExperiment2(Experiment):
        def tasks(self) -> dict[str, Task]:
            subexp = TestExperiment().tasks()
            subexp["final"] = Task(multiply, subexp["subresult"], subexp["multresult"], n="final")

            return subexp

    # print(task_graph)

    e = Executor.from_params(params={"type": "local"})  # MultithreadedLocalExecutor()
    ws = Workspace.from_params(params={"type": "memory"})
    # try:
    #     task_graph.result()
    # except:
    #     print("pass")

    exp = TestExperiment()
    print("run exp")
    exp.run(executor=e, workspace=ws)

    exp2 = TestExperiment2()
    print("run exp2")
    exp2.run(executor=e, workspace=ws)

    print(exp2.result(workspace=ws, step_name="subresult"))
    print(exp2.result(workspace=ws, step_name="multresult"))
    print(exp2.result(workspace=ws, step_name="final"))

    # f = task_graph.run(workspace=ws, executor=e)
    # print(f)
    # r = f.result()  # type: ignore
    # print(r)
    print(ws.d)

    # print(task_graph.result(ws, e))
