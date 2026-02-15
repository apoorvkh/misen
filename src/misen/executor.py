"""Executor interface for submitting a Task's DAG to an execution backend (e.g. a local process or SLURM).

Convention: Graph edge A -> B indicates that A depends on B.

Overview:
  1. The Task DAG is decomposed into WorkUnits (connected subgraphs), anchored at DAG roots and cacheable Tasks.
     Each WorkUnit contains the reachable subgraph of non-cacheable Tasks (truncated at downstream cacheable Tasks).
  2. WorkUnits are submitted to the backend for execution in dependency order.
     WorkUnits should assume their dependencies' roots are already cached (and can simply be retrieved) at runtime.
  3. Jobs are yielded and can be used to monitor the execution status of each WorkUnit.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Literal, TypeAlias, TypeVar, cast, get_args

from typing_extensions import assert_never

from misen.utils.settings import FromSettingsABC
from misen.utils.work_unit import WorkUnit, build_work_graph

if TYPE_CHECKING:
    from pathlib import Path

    from misen.tasks import Task
    from misen.utils.graph import DependencyGraph
    from misen.utils.snapshot import Snapshot
    from misen.workspace import Workspace

__all__ = ["Executor", "Job"]

ExecutorType: TypeAlias = Literal["local", "slurm"]
JobT = TypeVar("JobT", bound="Job")
SnapshotT = TypeVar("SnapshotT", bound="Snapshot")


class Executor(FromSettingsABC, Generic[JobT, SnapshotT]):
    """Abstract interface for implementing an Executor for a specific backend."""

    def submit(self, tasks: set[Task], workspace: Workspace) -> DependencyGraph[CompletedJob | JobT]:
        """Submit a set of tasks for execution. Tasks will be run by backend respecting dependency order.

        Returns:
            A dependency graph of backend-specific Job handles corresponding to WorkUnits.
        """
        work_graph: DependencyGraph[WorkUnit] = build_work_graph(tasks=tasks)

        # dispatch work units and collect job handles

        jobs: dict[WorkUnit, CompletedJob | JobT] = {}
        snapshot: SnapshotT | None = None

        for w in work_graph:  # dependency order
            if w.root.done(workspace=workspace):
                jobs[w] = CompletedJob(work_unit=w)
            else:
                # lazily snapshot only when at least one work unit actually needs dispatch
                if snapshot is None:
                    snapshot = self._make_snapshot(workspace=workspace)
                dependencies = {jobs[d] for d in w.dependencies if not isinstance(jobs[d], CompletedJob)}
                jobs[w] = self._dispatch(work_unit=w, dependencies=dependencies, workspace=workspace, snapshot=snapshot)

        # return job graph corresponding to work graph

        job_graph = cast("DependencyGraph[CompletedJob | JobT]", work_graph.copy())
        for i in job_graph.node_indices():
            job_graph[i] = jobs[work_graph[i]]

        return job_graph

    @abstractmethod
    def _make_snapshot(self, workspace: Workspace) -> SnapshotT:
        """Create (or reuse) an execution snapshot for this submit call."""

    @abstractmethod
    def _dispatch(
        self, work_unit: WorkUnit, dependencies: set[JobT], workspace: Workspace, snapshot: SnapshotT
    ) -> JobT:
        """Dispatch a WorkUnit to the backend. Will run `work_unit.execute(workspace)` after dependencies are completed.

        Args:
            work_unit: The WorkUnit to dispatch.
            dependencies: Job handles corresponding to prerequisite (incomplete) WorkUnits.
            workspace: Workspace providing Task artifact caching and retrieval.
            snapshot: Executor-specific environment snapshot.

        Returns:
            A Job handle that can be queried for execution state.
        """

    # Below: FromSettingsABC implementation. Permits initializing an Executor class from TOML settings or CLI.

    @staticmethod
    def _settings_key() -> str:
        """Return the TOML settings key for executor configuration."""
        return "executor"

    @staticmethod
    def _default() -> Executor:
        """Return the default executor implementation."""
        from misen.executors.local import LocalExecutor

        return LocalExecutor()

    @classmethod
    def _resolve_type(cls, type_name: str | ExecutorType) -> type[Executor]:
        """Resolve an executor type name to a class."""
        if type_name in get_args(ExecutorType):
            type_name = cast("ExecutorType", type_name)
            match type_name:
                case "local":
                    from misen.executors.local import LocalExecutor

                    return LocalExecutor
                case "slurm":
                    from misen.executors.slurm import SlurmExecutor

                    return SlurmExecutor
                case _:
                    assert_never(type_name)
        return super()._resolve_type(type_name)


class Job(ABC):
    """Abstract job handle returned by an executor."""

    __slots__ = ("job_id", "log_path", "work_unit")

    job_id: str | None
    log_path: Path | None
    work_unit: WorkUnit

    def __init__(self, work_unit: WorkUnit, job_id: str | None = None, log_path: Path | None = None) -> None:
        """Initialize the job wrapper for a work unit."""
        self.work_unit = work_unit
        self.job_id = job_id
        self.log_path = log_path

    @abstractmethod
    def state(self) -> Literal["pending", "running", "done", "failed", "unknown"]:
        """Return the current job state."""

    def wait(self, poll_s: float = 0.5) -> None:
        """Block until the job reaches a terminal state."""
        while True:
            if self.state() in ("done", "failed"):
                return
            time.sleep(poll_s)


class CompletedJob(Job):
    """Job placeholder for already-completed work units."""

    __slots__ = ()

    def __init__(self, work_unit: WorkUnit) -> None:
        """Initialize a completed job wrapper."""
        super().__init__(work_unit=work_unit, job_id=None, log_path=None)

    def state(self) -> Literal["done"]:
        """Return the completed state."""
        return "done"
