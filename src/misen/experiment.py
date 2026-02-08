"""Experiment definition and CLI helpers."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import make_dataclass
from functools import cache
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, cast

import tyro
from msgspec import Struct

from misen.executor import Executor, ExecutorType
from misen.task import Task
from misen.utils.auto import resolve_auto
from misen.utils.settings import DEFAULT_SETTINGS_FILE, Settings
from misen.workspace import Workspace, WorkspaceType

__all__ = ["Experiment"]


TasksT = TypeVar("TasksT", bound=Mapping[str, Task])


class Experiment(Struct, Generic[TasksT], frozen=True):
    """Base class for defining experiment task collections."""

    @abstractmethod
    def tasks(self) -> TasksT:
        """Return the mapping of task names to Task objects."""
        ...

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Cache the tasks method for subclasses."""
        super().__init_subclass__(**kwargs)
        # @cache downstream implementations of `tasks()`
        setattr(cls, "tasks", cache(cls.tasks))  # noqa: B010

    def __getitem__(self, key: str) -> Task:
        """Return a task by name."""
        return self.tasks()[key]

    def result(self, key: str, workspace: Workspace | Literal["auto"] = "auto") -> object:
        """Compute and return a result for the named task."""
        return self.tasks()[key].result(workspace=workspace)

    def run(
        self,
        workspace: Workspace | Literal["auto"] = "auto",
        executor: Executor | Literal["auto"] = "auto",
    ) -> None:
        """Submit all tasks in the experiment to an executor."""
        workspace = resolve_auto(workspace=workspace)
        executor = resolve_auto(executor=executor)
        executor.submit(tasks=set(self.tasks().values()), workspace=workspace)

    @classmethod
    def cli(cls) -> None:
        """Run a command-line interface for the experiment."""
        _fields_without_defaults = []
        _fields_with_defaults = [
            ("command", Literal["run", "count"], "run"),
            ("settings_file", Path, DEFAULT_SETTINGS_FILE),
            ("executor_type", ExecutorType | Literal["auto"], "auto"),
            ("workspace_type", WorkspaceType | Literal["auto"], "auto"),
        ]

        args, _ = tyro.cli(
            make_dataclass("", _fields_without_defaults + _fields_with_defaults),
            add_help=False,
            return_unknown_args=True,
        )
        args = cast("Any", args)  # TODO ?

        if args.executor_type != "auto":
            _fields_without_defaults.append(("executor", Executor.resolve_type(args.executor_type)))
        if args.workspace_type != "auto":
            _fields_without_defaults.append(("workspace", Workspace.resolve_type(args.workspace_type)))
        _fields_without_defaults.append(("experiment", tyro.conf.OmitArgPrefixes[cls]))  # ty:ignore[invalid-type-form]  # TODO

        args = tyro.cli(make_dataclass("", _fields_without_defaults + _fields_with_defaults))
        args = cast("Any", args)  # TODO ?

        settings = Settings(file=args.settings_file)

        executor = Executor.auto(settings=settings) if args.executor_type == "auto" else args.executor

        workspace = Workspace.auto(settings=settings) if args.workspace_type == "auto" else args.workspace

        match args.command:
            case "run":
                args.experiment.run(executor=executor, workspace=workspace)
