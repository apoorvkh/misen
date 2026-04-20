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

import logging
from abc import abstractmethod
from collections.abc import Mapping
from functools import cache
from typing import Any, Generic, Literal

from msgspec import Struct
from typing_extensions import TypeVar

from misen.executor import Executor
from misen.tasks import Task
from misen.utils.cli.experiment import _ClassOrInstanceMethod, experiment_cli
from misen.workspace import Workspace

__all__ = ["Experiment"]


TasksT = TypeVar("TasksT", bound=Mapping[str, Task], default=Mapping[str, Task[Any]])
logger = logging.getLogger(__name__)


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
        # Memoize tasks() per experiment instance so DAG construction is deterministic
        # and repeated access (run/result/CLI) does not rebuild task objects.
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
        workspace = Workspace.resolve_auto(workspace)
        executor = Executor.resolve_auto(executor)
        experiment_tasks = set(self.tasks().values())
        logger.info(
            "Running experiment %s with %d task(s) using executor=%s workspace=%s.",
            self.__class__.__name__,
            len(experiment_tasks),
            executor.__class__.__name__,
            workspace.__class__.__name__,
        )
        executor.submit(tasks=experiment_tasks, workspace=workspace)

    @_ClassOrInstanceMethod
    def cli(self: Any) -> None:
        """Run the generated command-line interface for this experiment.

        Class access (``TrainingExperiment.cli()``) exposes field defaults.
        Instance access (``TrainingExperiment(lr=0.1).cli()``) seeds CLI defaults
        from the instance's bound field values while still allowing command-line
        overrides. (``self`` here is the class on class access, the instance on
        instance access — see :class:`_ClassOrInstanceMethod`.)
        """
        experiment_cli(self)
