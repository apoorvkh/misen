from __future__ import annotations

from abc import abstractmethod
from ast import literal_eval
from functools import cache
from typing import Annotated, Generic, Literal, Mapping, TypeVar

import tyro
from msgspec import Struct
from typing_extensions import Self

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

    def __getitem__(self, key: str) -> Task:
        return self.tasks()[key]

    def result(self, key: str, workspace: Workspace | None = None) -> object:
        return self.tasks()[key].result(workspace=workspace)

    def run(self, workspace: Workspace | None, executor: Executor | None) -> None:
        Task((lambda **kwargs: None), **self.tasks()).run(workspace=workspace, executor=executor)

    @classmethod
    def _cli_entrypoint(
        cls,
        executor_params: Annotated[str | None, tyro.conf.arg(name="executor", default=None)],
        workspace_params: Annotated[str | None, tyro.conf.arg(name="workspace", default=None)],
        experiment: tyro.conf.OmitArgPrefixes[Self],
        command: tyro.conf.Positional[Literal["run", "count"]] = "run",
    ):
        if executor_params is None:
            executor = Executor.default()
        else:
            executor = Executor.from_params(literal_eval(executor_params))

        if workspace_params is None:
            workspace = Workspace.default()
        else:
            workspace = Workspace.from_params(literal_eval(workspace_params))

        match command:
            case "run":
                experiment.run(workspace=workspace, executor=executor)

    @classmethod
    def cli(cls):
        tyro.cli(cls._cli_entrypoint)
