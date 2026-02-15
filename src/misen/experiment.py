"""Experiment-level orchestration primitives.

``Experiment`` provides a thin, declarative wrapper around a named collection of
tasks. It keeps execution concerns modular:

- Task graph semantics stay in :mod:`misen.tasks`.
- Backend dispatch stays in :mod:`misen.executor`.
- Artifact/caching policy stays in :mod:`misen.workspace`.

The experiment layer only binds those pieces together and exposes convenience
entry points (`run`, `result`, `cli`) for end users.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from functools import cache
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from msgspec import Struct

from misen.tasks import Task
from misen.utils.auto import resolve_auto
from misen.utils.experiment_cli import experiment_cli

if TYPE_CHECKING:
    from misen.executor import Executor
    from misen.workspace import Workspace

__all__ = ["Experiment"]


TasksT = TypeVar("TasksT", bound=Mapping[str, Task])


class Experiment(Struct, Generic[TasksT], frozen=True):
    """Base class for defining a named task collection.

    Subclasses implement :meth:`tasks` and return a stable mapping from user
    names (for example ``"train"`` or ``"eval"``) to :class:`misen.tasks.Task`
    instances.
    """

    @abstractmethod
    def tasks(self) -> TasksT:
        """Return the experiment's named task mapping.

        Returns:
            Mapping from logical task names to task objects.
        """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap subclass ``tasks()`` in a per-instance cache.

        Args:
            **kwargs: Standard subclass construction kwargs.
        """
        super().__init_subclass__(**kwargs)
        # @cache downstream implementations of `tasks()`
        setattr(cls, "tasks", cache(cls.tasks))  # noqa: B010

    def __getitem__(self, key: str) -> Task:
        """Return a task from the named task mapping.

        Args:
            key: Name previously returned by :meth:`tasks`.

        Returns:
            The associated task object.
        """
        return self.tasks()[key]

    def result(self, key: str, workspace: Workspace | Literal["auto"] = "auto") -> object:
        """Compute or load the result for one named task.

        Args:
            key: Task name in :meth:`tasks`.
            workspace: Workspace instance, or ``"auto"`` to resolve from
                settings/defaults.

        Returns:
            Result object returned by the underlying task.
        """
        return self.tasks()[key].result(workspace=workspace)

    def run(
        self,
        workspace: Workspace | Literal["auto"] = "auto",
        executor: Executor | Literal["auto"] = "auto",
    ) -> None:
        """Submit all experiment tasks to an executor.

        Args:
            workspace: Workspace instance, or ``"auto"`` to resolve from
                settings/defaults.
            executor: Executor instance, or ``"auto"`` to resolve from
                settings/defaults.
        """
        workspace = resolve_auto(workspace=workspace)
        executor = resolve_auto(executor=executor)
        executor.submit(tasks=set(self.tasks().values()), workspace=workspace)

    @classmethod
    def cli(cls) -> None:
        """Run the generated command-line interface for this experiment class."""
        experiment_cli(cls)
