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
import weakref
from abc import abstractmethod
from collections.abc import Collection, Mapping
from functools import wraps
from typing import Any, Generic, Literal, TypeAlias, cast

from msgspec import Struct, StructMeta
from typing_extensions import TypeVar

from misen.executor import Executor
from misen.tasks import Task
from misen.utils.cli.experiment import _ClassOrInstanceMethod, experiment_cli
from misen.workspace import Workspace

__all__ = ["Experiment"]


TasksT = TypeVar("TasksT", bound=Mapping[str, Task], default=Mapping[str, Task[Any]])
_ExperimentTasks: TypeAlias = Mapping[str, Task[Any]] | Collection[Task[Any]]
logger = logging.getLogger(__name__)
_TASKS_CACHE: weakref.WeakKeyDictionary[Experiment[Any], _ExperimentTasks] = weakref.WeakKeyDictionary()


class _FrozenStructMeta(StructMeta):
    def __new__(
        mcls: type[_FrozenStructMeta],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **struct_config: Any,
    ) -> _FrozenStructMeta:
        struct_config.setdefault("frozen", True)
        # WeakKeyDictionary keys must support weak references.
        struct_config.setdefault("weakref", True)
        return super().__new__(mcls, name, bases, namespace, **struct_config)


class Experiment(Struct, Generic[TasksT], metaclass=_FrozenStructMeta):
    """Base class for defining an experiment's task collection.

    Subclasses implement :meth:`tasks` returning either:

    - a ``Mapping[str, Task]`` of named tasks — keys are user-facing labels
      usable via :meth:`__getitem__`, :meth:`result`, and the ``--task`` CLI
      flag. Parameterize with ``Experiment[MyTasksDict]`` to preserve per-key
      typing through ``.tasks()``.
    - a ``Collection[Task]`` (set/list/tuple) of unkeyed tasks — no named
      access; use hash-prefix lookup in the CLI if individual selection is
      needed.

    Internal code paths (``run``, CLI, TUI) read the normalized mapping view
    via :meth:`_tasks`, so ``tasks()`` preserves the subclass's declared
    return type verbatim for static analysis.
    """

    @abstractmethod
    def tasks(self) -> TasksT | Collection[Task[Any]]:
        """Return the experiment's tasks.

        Returns:
            Either a ``Mapping[str, Task]`` for named access (the typed shape
            via ``TasksT`` when the experiment is parameterized), or a
            ``Collection[Task]`` (set, list, tuple) for an unkeyed union.
        """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Cache :meth:`tasks` to avoid recomputing for the same (frozen) instance."""
        super().__init_subclass__(**kwargs)
        _tasks_fn = cls.tasks

        @wraps(_tasks_fn)
        def cached_tasks_fn(self: Experiment) -> _ExperimentTasks:
            cached = _TASKS_CACHE.get(self)
            if cached is None:
                cached = _tasks_fn(self)
                _TASKS_CACHE[self] = cached
            return cached

        type.__setattr__(cls, "tasks", cached_tasks_fn)

    def normalized_tasks(self) -> Mapping[str, Task[Any]]:
        """Return a normalized ``Mapping`` view of :meth:`tasks`.

        Mapping returns pass through unchanged; ``Collection`` returns are
        keyed by each task's ``task_hash().b32()`` so keys are stable and
        unique by construction. Internal code (``run``, CLI, TUI) goes
        through this view so it can always assume a mapping.
        """
        raw = self.tasks()
        if not isinstance(raw, Mapping):
            return {task.task_hash().b32(): task for task in raw}
        return cast("Mapping[str, Task[Any]]", raw)

    def __getitem__(self, key: str) -> Task:
        """Return a task from the named task mapping.

        Args:
            key: Name previously returned by :meth:`tasks`.

        Returns:
            The associated task object.
        """
        return self.normalized_tasks()[key]

    def result(self, key: str, workspace: Workspace | Literal["auto"] = "auto") -> object:
        """Compute or load the result for one named task.

        Args:
            key: Task name in :meth:`tasks`.
            workspace: Workspace instance, or ``"auto"`` to resolve from
                settings/defaults.

        Returns:
            Result object returned by the underlying task.
        """
        return self.normalized_tasks()[key].result(workspace=workspace)

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
        experiment_tasks = set(self.normalized_tasks().values())
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
