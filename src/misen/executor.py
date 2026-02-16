"""Executor abstraction and job lifecycle model.

Design overview:

1. ``Task`` graphs are decomposed into cache-bounded :class:`misen.utils.work_unit.WorkUnit`
   nodes. This keeps scheduling granularity aligned with caching boundaries.
2. Executors submit work units in dependency order, but may run independent
   units concurrently according to backend policy (local scheduler, SLURM, etc.).
3. Backends expose lightweight :class:`Job` handles for polling and waiting.

This module intentionally does not encode backend-specific logic. Concrete
behavior lives in backend modules under :mod:`misen.executors`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Literal, TypeAlias, TypeVar, cast, get_args

from typing_extensions import assert_never

from misen.utils.runtime_events import runtime_activity, runtime_event, runtime_progress, work_unit_label
from misen.utils.settings import FromSettingsABC
from misen.utils.work_unit import build_work_graph

if TYPE_CHECKING:
    from pathlib import Path

    from misen.tasks import Task
    from misen.utils.graph import DependencyGraph
    from misen.utils.snapshot import Snapshot
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = ["Executor", "Job"]

ExecutorType: TypeAlias = Literal["local", "in_process", "slurm"]
JobT = TypeVar("JobT", bound="Job")
SnapshotT = TypeVar("SnapshotT", bound="Snapshot")


class Executor(FromSettingsABC, Generic[JobT, SnapshotT]):
    """Abstract execution backend interface.

    Subclasses provide snapshot creation and dispatch behavior; shared submission
    logic here handles dependency-aware graph traversal and completed-work short
    circuiting.
    """

    def submit(self, tasks: set[Task], workspace: Workspace) -> DependencyGraph[CompletedJob | JobT]:
        """Submit tasks for execution on this backend.

        The method first converts task DAGs into a work-unit DAG. Work units
        already marked done in the workspace are represented as
        :class:`CompletedJob` placeholders and skipped.

        Args:
            tasks: Root tasks requested by the caller.
            workspace: Workspace used for cache inspection and artifact access.

        Returns:
            Dependency graph of job handles keyed to work-unit topology.
        """
        work_graph: DependencyGraph[WorkUnit] = build_work_graph(tasks=tasks)
        work_units = list(work_graph)  # dependency order

        jobs: dict[WorkUnit, CompletedJob | JobT] = {
            w: CompletedJob(work_unit=w) for w in work_units if w.done(workspace=workspace)
        }

        num_complete = len(jobs)
        num_dispatch = len(work_units) - num_complete
        executor_name = self.__class__.__name__

        # dispatch work units and collect job handles

        if num_dispatch > 0:
            started_at = time.perf_counter()
            try:
                with runtime_activity("Creating a snapshot of the project environment", style="yellow"):
                    snapshot = self._make_snapshot(workspace=workspace)
            except Exception:
                elapsed_s = time.perf_counter() - started_at
                runtime_event(
                    f"Failed to create a snapshot of the project environment in {elapsed_s:.2f}s)", style="bold red"
                )
                raise
            elapsed_s = time.perf_counter() - started_at
            runtime_event(f"Created a snapshot of the project environment in {elapsed_s:.2f}s", style="green")

            with runtime_progress(f"Submitting work units to {executor_name}", total=num_dispatch) as progress_bar:
                for w in work_units:
                    if w in jobs:
                        continue

                    started_at = time.perf_counter()
                    try:
                        dependencies = {jobs[d] for d in w.dependencies if not isinstance(jobs[d], CompletedJob)}
                        jobs[w] = self._dispatch(
                            work_unit=w,
                            dependencies=dependencies,
                            workspace=workspace,
                            snapshot=snapshot,
                        )
                    except Exception:
                        elapsed_s = time.perf_counter() - started_at
                        runtime_event(
                            (f"Dispatch failed for {work_unit_label(w)} in {elapsed_s:.2f}s"), style="bold red"
                        )
                        raise
                    progress_bar(1)

        runtime_event(
            (
                f"Submitted {num_dispatch} work unit(s) to {executor_name}"
                f"{f' ({num_complete} already complete)' if num_complete > 0 else ''}"
            ),
            style="green bold",
        )

        # return job graph corresponding to work graph

        job_graph = cast("DependencyGraph[CompletedJob | JobT]", work_graph.copy())
        for i in job_graph.node_indices():
            job_graph[i] = jobs[work_graph[i]]

        return job_graph

    @abstractmethod
    def _make_snapshot(self, workspace: Workspace) -> SnapshotT:
        """Create or reuse an execution snapshot for a submit call.

        Args:
            workspace: Workspace used to materialize snapshot artifacts.

        Returns:
            Backend-specific snapshot object.
        """

    @abstractmethod
    def _dispatch(
        self, work_unit: WorkUnit, dependencies: set[JobT], workspace: Workspace, snapshot: SnapshotT
    ) -> JobT:
        """Dispatch a work unit once dependency jobs are satisfied.

        Args:
            work_unit: Work unit to execute.
            dependencies: Job handles corresponding to prerequisite (incomplete) WorkUnits.
            workspace: Workspace providing Task artifact caching and retrieval.
            snapshot: Executor-specific environment snapshot.

        Returns:
            A Job handle that can be queried for execution state.
        """

    # Below: FromSettingsABC implementation. Permits initializing an Executor class from TOML settings or CLI.

    @staticmethod
    def _settings_key() -> str:
        """Return TOML key used for executor auto-configuration."""
        return "executor"

    @staticmethod
    def _default() -> Executor:
        """Return default executor implementation."""
        from misen.executors.local import LocalExecutor

        return LocalExecutor()

    @classmethod
    def _resolve_type(cls, type_name: str | ExecutorType) -> type[Executor]:
        """Resolve an executor type string to a concrete class.

        Args:
            type_name: Built-in short name or ``module:Class`` string.

        Returns:
            Resolved executor class.
        """
        if type_name in get_args(ExecutorType):
            type_name = cast("ExecutorType", type_name)
            match type_name:
                case "local":
                    from misen.executors.local import LocalExecutor

                    return LocalExecutor
                case "in_process":
                    from misen.executors.in_process import InProcessExecutor

                    return InProcessExecutor
                case "slurm":
                    from misen.executors.slurm import SlurmExecutor

                    return SlurmExecutor
                case _:
                    assert_never(type_name)
        return super()._resolve_type(type_name)


class Job(ABC):
    """Abstract job handle returned by an executor backend."""

    __slots__ = ("job_id", "log_path", "work_unit")

    job_id: str | None
    log_path: Path | None
    work_unit: WorkUnit

    def __init__(self, work_unit: WorkUnit, job_id: str | None = None, log_path: Path | None = None) -> None:
        """Initialize a job handle.

        Args:
            work_unit: Work unit associated with this job.
            job_id: Backend-facing job identifier, if available.
            log_path: Optional path to job-level logs.
        """
        self.work_unit = work_unit
        self.job_id = job_id
        self.log_path = log_path

    @abstractmethod
    def state(self) -> Literal["pending", "running", "done", "failed", "unknown"]:
        """Return the current backend job state."""

    def wait(self, poll_s: float = 0.5) -> None:
        """Block until job reaches a terminal state.

        Args:
            poll_s: Polling interval in seconds.
        """
        while True:
            if self.state() in ("done", "failed"):
                return
            time.sleep(poll_s)


class CompletedJob(Job):
    """Placeholder job for work units that are already complete in cache."""

    __slots__ = ()

    def __init__(self, work_unit: WorkUnit) -> None:
        """Initialize a completed-job wrapper.

        Args:
            work_unit: Completed work unit.
        """
        super().__init__(work_unit=work_unit, job_id=None, log_path=None)

    def state(self) -> Literal["done"]:
        """Return terminal ``done`` state."""
        return "done"
