from __future__ import annotations

from typing import TypedDict

import msgspec

from misen import Settings, Task, task
from misen.experiment import Experiment
from misen.workspace import Workspace, WorkspaceConfig
from misen.workspaces.memory import MemoryWorkspace, MemoryWorkspaceConfig


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
    workspace = MemoryWorkspace(config=MemoryWorkspaceConfig(i=10))
    print(workspace)

    workspace = MemoryWorkspaceConfig(i=20).load()
    print(workspace)


    # workspace = WorkspaceConfig(type="memory").load()
    # print(workspace)

    workspace = WorkspaceConfig().load()
    print(workspace)

    # workspace = WorkspaceConfig().from_settings(settings=Settings())
    # print(workspace)

    # TimesTwoPlusTen.cli()
    # print(msgspec.to_builtins(cfg, order="sorted"))

    # print(MemoryWorkspaceConfig(i=10).load())
    # print(MemoryWorkspaceConfig(i=10).load() is MemoryWorkspace(config=MemoryWorkspaceConfig(i=10)))
    # print(WorkspaceConfig(type="misen.workspaces.memory:MemoryWorkspace").load() is MemoryWorkspace(config=MemoryWorkspaceConfig(i=10)))
    # experiment = TimesTwoPlusTen(x=5)
    # print(experiment.tasks()['first'].result())
    # print(experiment.tasks()['second'].result())
    # print(experiment.tasks()['third'].result())
