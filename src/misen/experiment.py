from __future__ import annotations

import sys
from abc import abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import Generic, Literal, Mapping, TypeVar

import tyro
from msgspec import Struct

from .executor import Executor, ExecutorConfig
from .settings import Settings  # noqa: TC001
from .task import Task
from .workspace import Workspace, WorkspaceConfig

TasksT = TypeVar("TasksT", bound=Mapping[str, Task])

ExecutorConfigT = TypeVar("ExecutorConfigT", bound=ExecutorConfig)
WorkspaceConfigT = TypeVar("WorkspaceConfigT", bound=WorkspaceConfig)
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
            executor: ExecutorConfig
            workspace: WorkspaceConfig

        base_args, _ = tyro.cli(
            BaseArgs,
            args=[arg for arg in sys.argv if arg != "--help"],
            return_unknown_args=True,
        )

        @dataclass
        class Args(Generic[ExperimentT, ExecutorConfigT, WorkspaceConfigT]):
            settings: Settings
            experiment: tyro.conf.OmitArgPrefixes[ExperimentT]
            executor: ExecutorConfigT
            workspace: WorkspaceConfigT
            command: Literal["run", "count"] = "run"

        args = tyro.cli(
            Args[
                cls,
                (
                    base_args.executor.resolve_config_type()
                    if base_args.executor.type != "auto"
                    else ExecutorConfig
                ),
                (
                    base_args.workspace.resolve_config_type()
                    if base_args.workspace.type != "auto"
                    else WorkspaceConfig
                ),
            ]
        )

        executor = args.executor.load(settings=args.settings)
        workspace = args.workspace.load(settings=args.settings)

        match args.command:
            case "run":
                args.experiment.run(workspace=workspace, executor=executor)
