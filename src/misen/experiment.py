from __future__ import annotations

import sys
from abc import abstractmethod
from dataclasses import make_dataclass
from functools import cache
from pathlib import Path
from typing import Generic, Literal, Mapping, TypeVar

import tyro
from msgspec import Struct

from .executor import Executor, ExecutorType
from .task import Task
from .utils.settings import DEFAULT_SETTINGS_FILE, Settings
from .workspace import Workspace, WorkspaceType

__all__ = ["Experiment"]


TasksT = TypeVar("TasksT", bound=Mapping[str, Task])

ExecutorT = TypeVar("ExecutorT", bound=Executor)
WorkspaceT = TypeVar("WorkspaceT", bound=Workspace)
ExperimentT = TypeVar("ExperimentT", bound="Experiment")


class Experiment(Generic[TasksT], Struct, frozen=True):
    @abstractmethod
    def tasks(self) -> TasksT: ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        setattr(cls, "tasks", cache(cls.tasks))

    def __getitem__(self, key: str) -> Task:
        return self.tasks()[key]

    def result(self, key: str, workspace: Workspace | None = None) -> object:
        return self.tasks()[key].result(workspace=workspace)

    def run(self, workspace: Workspace | None = None, executor: Executor | None = None) -> None:
        # TODO: note empty lambda task is submitted as well
        Task((lambda **kwargs: None), **self.tasks()).run(workspace=workspace, executor=executor)

    @classmethod
    def cli(cls):
        _fields_without_defaults = []
        _fields_with_defaults = [
            ("command", Literal["run", "count"], "run"),
            ("settings_file", Path, DEFAULT_SETTINGS_FILE),
            ("executor_type", ExecutorType, "auto"),
            ("workspace_type", WorkspaceType, "auto"),
        ]

        args, _ = tyro.cli(
            make_dataclass("", _fields_without_defaults + _fields_with_defaults),
            args=[arg for arg in sys.argv if arg != "--help"],
            return_unknown_args=True,
        )

        if args.executor_type != "auto":
            _fields_without_defaults.append(("executor", Executor._resolve_type(args.executor_type)))
        if args.workspace_type != "auto":
            _fields_without_defaults.append(("workspace", Workspace._resolve_type(args.workspace_type)))
        _fields_without_defaults.append(("experiment", tyro.conf.OmitArgPrefixes[cls]))

        args = tyro.cli(make_dataclass("", _fields_without_defaults + _fields_with_defaults))

        settings = Settings(file=args.settings_file)

        if args.executor_type == "auto":
            executor = Executor.auto(settings=settings)
        else:
            executor = args.executor

        if args.workspace_type == "auto":
            workspace = Workspace.auto(settings=settings)
        else:
            workspace = args.workspace

        match args.command:
            case "run":
                args.experiment.run(executor=executor, workspace=workspace)
