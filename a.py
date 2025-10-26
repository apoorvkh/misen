from __future__ import annotations

from typing import TypedDict

from misen import Task, task
from misen.executors.local import LocalExecutor
from misen.experiment import Experiment
from misen.workspaces.disk import DiskWorkspace


@task(cache_result=True)
def add(i: int, j: int) -> int:
    return i + j


@task(cache_result=True)
def multiply(i: int, j: int) -> int:
    return i * j


@task(cache_result=True)
def mod(i: int, j: int) -> int:
    return i % j


class ReturnType(TypedDict):
    first: Task[int]
    second: Task[int]
    third: Task[int]


class TimesTwoPlusTen(Experiment, frozen=True):
    x: int

    def tasks(self) -> ReturnType:
        first_step = Task(add, self.x, 10)
        second_step = Task(multiply, first_step.T, 2)
        third_step = Task(multiply, 15, 2)
        return ReturnType(
            first=first_step,
            second=second_step,
            third=third_step,
        )


if __name__ == "__main__":
    workspace = DiskWorkspace(directory="./")
    executor = LocalExecutor()

    exp = TimesTwoPlusTen(2)
    exp.run(workspace=workspace, executor=executor)
    print(exp.result("second"))
