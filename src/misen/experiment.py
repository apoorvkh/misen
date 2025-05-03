from __future__ import annotations

from abc import abstractmethod
from functools import cache
from typing import Annotated, Generic, Mapping, TypeVar

import tyro
from msgspec import Struct

from .executor import Executor
from .task import Task
from .workspace import Workspace

TasksT = TypeVar("TasksT", bound=Mapping[str, Task])


class Experiment(Generic[TasksT], Struct, frozen=True):
    @abstractmethod
    def tasks(self) -> TasksT:
        raise NotImplementedError(f"{self.__class__.__name__} must implement a tasks() method.")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        setattr(cls, "tasks", cache(cls.tasks))

    def run(self, workspace: Workspace | None, executor: Executor | None) -> None:
        Task((lambda **kwargs: None), **self.tasks()).run(workspace=workspace, executor=executor)

    @classmethod
    def cli(cls):
        executor, workspace, experiment = tyro.cli(
            tuple[
                Annotated[Executor, tyro.conf.arg(name="executor")],
                Annotated[Workspace, tyro.conf.arg(name="workspace")],
                Annotated[cls, tyro.conf.OmitArgPrefixes, tyro.conf.arg(name="experiment")],
            ]
        )

        experiment.run(workspace=workspace, executor=executor)
