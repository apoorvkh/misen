from __future__ import annotations

import sys
from abc import abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import Generic, Literal, Mapping, TypeVar

import tyro
from msgspec import Struct

from .executor import Executor
from .settings import Settings  # noqa: TC001
from .task import Task
from .workspace import Workspace

TasksT = TypeVar("TasksT", bound=Mapping[str, Task])

ExecutorT = TypeVar("ExecutorT", bound=Executor)
WorkspaceT = TypeVar("WorkspaceT", bound=Workspace)
ExperimentT = TypeVar("ExperimentT", bound="Experiment")


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

    def run(self, workspace: Workspace | None = None, executor: Executor | None = None) -> None:
        Task((lambda **kwargs: None), **self.tasks()).run(workspace=workspace, executor=executor)

    @classmethod
    def cli(cls):
        @dataclass
        class Args(Generic[ExecutorT, WorkspaceT, ExperimentT]):
            settings: Settings
            executor: ExecutorT
            workspace: WorkspaceT
            experiment: tyro.conf.OmitArgPrefixes[ExperimentT]
            command: Literal["run", "count"] = "run"

        args, _ = tyro.cli(
            Args[Executor, Workspace, cls],
            args=[arg for arg in sys.argv if arg != "--help"],
            return_unknown_args=True,
        )
        args = tyro.cli(
            Args[
                args.executor._resolve_type() or Executor,
                args.workspace._resolve_type() or Workspace,
                cls,
            ]
        )

        if type(args.executor) is Executor:
            args.executor = Executor.load(settings=args.settings)

        if type(args.workspace) is Workspace:
            args.workspace = Workspace.load(settings=args.settings)

        match args.command:
            case "run":
                args.experiment.run(workspace=args.workspace, executor=args.executor)
