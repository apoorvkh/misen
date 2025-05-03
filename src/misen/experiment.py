from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, Mapping, TypeVar

import tyro
from msgspec import Struct

from .executor import Executor
from .task import Task
from .utils.cached_property import cached_property
from .workspace import Workspace

TasksT = TypeVar("TasksT", bound=Mapping[str, Task])


class Experiment(Generic[TasksT], Struct, frozen=True):
    @abstractmethod
    def tasks(self) -> TasksT:
        raise NotImplementedError

    def __getattribute__(self, name: str) -> Any:
        if name == "tasks":
            return cached_property(self, type(self).tasks, key="__cached_task_graph__")
        return super().__getattribute__(name)

    def run(self, workspace: Workspace | None, executor: Executor | None) -> None:
        Task((lambda **kwargs: None), **self.tasks()).run(
            workspace=workspace, executor=executor
        )

    @classmethod
    def _cli_run(
        cls,
        workspace: Workspace | None = None,
        executor: Executor | None = None,
    ) -> None:
        cls().run(workspace=workspace, executor=executor)

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
