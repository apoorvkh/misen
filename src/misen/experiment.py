from __future__ import annotations

import sys
from abc import abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import Generic, Literal, Mapping, TypeVar

import tyro
from msgspec import Struct

from .executor import Executor, ExecutorType
from .settings import Settings  # noqa: TC001
from .task import Task
from .workspace import Workspace, WorkspaceType

TasksT = TypeVar("TasksT", bound=Mapping[str, Task])

ExecutorT = TypeVar("ExecutorT", bound=Executor)
WorkspaceT = TypeVar("WorkspaceT", bound=Workspace)
ExperimentT = TypeVar("ExperimentT", bound="Experiment")


class Experiment(Generic[TasksT], Struct, frozen=True):
    @abstractmethod
    def tasks(self) -> TasksT:
        raise NotImplementedError

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
        class BaseArgs:
            settings: Settings
            executor_type: ExecutorType = "auto"
            workspace_type: WorkspaceType = "auto"

        base_args, _ = tyro.cli(
            BaseArgs,
            args=[arg for arg in sys.argv if arg != "--help"],
            return_unknown_args=True,
        )

        executor_cls = Executor.resolve_type(base_args.executor_type)
        if executor_cls is Executor:
            executor_cls = tyro.conf.Suppress[Executor]

        workspace_cls = Workspace.resolve_type(base_args.workspace_type)
        if workspace_cls is Workspace:
            workspace_cls = tyro.conf.Suppress[Workspace]

        @dataclass
        class Args(Generic[ExperimentT, ExecutorT, WorkspaceT]):
            settings: Settings
            experiment: tyro.conf.OmitArgPrefixes[ExperimentT]
            executor: ExecutorT
            workspace: WorkspaceT
            workspace_type: WorkspaceType = "auto"
            executor_type: ExecutorType = "auto"
            command: Literal["run", "count"] = "run"

        args = tyro.cli(Args[cls, executor_cls, workspace_cls])

        if args.executor_type == "auto":
            executor = Executor.auto(settings=args.settings)
        else:
            executor = args.executor

        if args.workspace_type == "auto":
            workspace = Workspace.auto(settings=args.settings)
        else:
            workspace = args.workspace

        match args.command:
            case "run":
                args.experiment.run(executor=executor, workspace=workspace)
