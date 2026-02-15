"""Experiment definition and CLI helpers."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from functools import cache
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from msgspec import Struct

from misen.task import Task
from misen.utils.auto import resolve_auto
from misen.utils.experiment_cli import experiment_cli

if TYPE_CHECKING:
    from misen.executor import Executor
    from misen.workspace import Workspace

__all__ = ["Experiment"]


TasksT = TypeVar("TasksT", bound=Mapping[str, Task])


class Experiment(Struct, Generic[TasksT], frozen=True):
    """Base class for defining experiment task collections."""

    @abstractmethod
    def tasks(self) -> TasksT:
        """Return the mapping of task names to Task objects."""

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
        experiment_cli(cls)
