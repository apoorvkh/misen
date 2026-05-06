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

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Generic, Literal, TypeAlias, TypeVar, cast

from misen.utils.runtime_events import runtime_activity, runtime_event, runtime_progress, task_label, work_unit_label
from misen.utils.settings import Configurable
from misen.utils.snapshot import LocalSnapshot
from misen.utils.work_unit import build_work_graph

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from misen.task_metadata import Resources
    from misen.tasks import Task
    from misen.utils.graph import DependencyGraph
    from misen.utils.snapshot import Snapshot
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = ["Executor", "Job", "JobState", "bulk_job_states"]

ExecutorType: TypeAlias = Literal["local", "in_process", "slurm", "skypilot"]
JobState: TypeAlias = Literal["pending", "running", "done", "failed", "unknown"]
_VALID_JOB_STATES: frozenset[JobState] = frozenset({"pending", "running", "done", "failed", "unknown"})
JobT = TypeVar("JobT", bound="Job")
SnapshotT = TypeVar("SnapshotT", bound="Snapshot")
logger = logging.getLogger(__name__)


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    """Return ``singular`` when ``count == 1`` else ``plural`` (default: ``singular + 's'``)."""
    return singular if count == 1 else (plural if plural is not None else f"{singular}s")


class Executor(Configurable, Generic[JobT, SnapshotT]):
    """Abstract execution backend interface.

    Subclasses provide snapshot creation and dispatch behavior; shared submission
    logic here handles dependency-aware graph traversal and completed-work short
    circuiting.
    """

    _config_key: ClassVar[str] = "executor"
    _config_default_type: ClassVar[str] = "misen.executors.local:LocalExecutor"
    _config_aliases: ClassVar[dict[ExecutorType, str]] = {
        "local": "misen.executors.local:LocalExecutor",
        "in_process": "misen.executors.in_process:InProcessExecutor",
        "slurm": "misen.executors.slurm:SlurmExecutor",
        "skypilot": "misen.executors.skypilot:SkyPilotExecutor",
    }

    def submit(
        self,
        tasks: set[Task],
        workspace: Workspace,
        *,
        blocking: bool = False,
    ) -> tuple[DependencyGraph[CompletedJob | JobT], SnapshotT | None]:
        """Submit tasks for execution on this backend.

        The method first converts task DAGs into a work-unit DAG. Work units
        already marked done in the workspace are represented as
        :class:`CompletedJob` placeholders and skipped.

        Args:
            tasks: Root tasks requested by the caller.
            workspace: Workspace used for cache inspection and artifact access.
            blocking: Whether to wait until all submitted jobs reach a
                terminal state before returning.

        Returns:
            Tuple of (dependency graph of job handles, snapshot used for
            dispatched work units or ``None`` if all work was cached).
        """
        work_graph: DependencyGraph[WorkUnit] = build_work_graph(tasks=tasks)
        work_units = list(work_graph)
        executor_name = self.__class__.__name__
        logger.info(
            "%s received %d root task(s); built %d work unit(s).",
            executor_name,
            len(tasks),
            len(work_units),
        )

        jobs: dict[WorkUnit, CompletedJob | JobT] = {
            w: CompletedJob(work_unit=w) for w in work_units if w.done(workspace=workspace)
        }
        pending_work_units = [work_unit for work_unit in work_units if work_unit not in jobs]

        num_complete = len(jobs)
        num_dispatch = len(pending_work_units)
        logger.debug(
            "%s found %d complete work unit(s) and %d pending work unit(s).",
            executor_name,
            num_complete,
            num_dispatch,
        )

        snapshot: SnapshotT | None = None
        if pending_work_units:
            logger.info("%s creating snapshot for %d pending work unit(s).", executor_name, num_dispatch)
            started_at = time.perf_counter()
            try:
                with runtime_activity("Creating a snapshot of the project environment", style="yellow"):
                    snapshot = self._make_snapshot(workspace=workspace)
            except Exception:
                elapsed_s = time.perf_counter() - started_at
                logger.exception("%s failed to create a snapshot after %.2fs.", executor_name, elapsed_s)
                runtime_event(
                    f"Failed to create a snapshot of the project environment in {elapsed_s:.2f}s", style="bold red"
                )
                raise
            elapsed_s = time.perf_counter() - started_at
            logger.info("%s created snapshot in %.2fs.", executor_name, elapsed_s)
            runtime_event(f"Created a snapshot of the project environment in {elapsed_s:.2f}s", style="green")

            with runtime_progress(f"Submitting jobs to {executor_name}", total=num_dispatch) as progress_bar:
                for work_unit in pending_work_units:
                    started_at = time.perf_counter()
                    try:
                        dependencies = {
                            jobs[dependency]
                            for dependency in work_unit.dependencies
                            if not isinstance(jobs[dependency], CompletedJob)
                        }
                        logger.debug(
                            "%s dispatching %s with %d dependency job(s).",
                            executor_name,
                            work_unit_label(work_unit),
                            len(dependencies),
                        )
                        dispatched_job = self._dispatch(
                            work_unit=work_unit,
                            dependencies=dependencies,
                            workspace=workspace,
                            snapshot=snapshot,
                        )
                        jobs[work_unit] = dispatched_job
                        logger.debug(
                            "%s dispatched %s (job_id=%s) in %.2fs.",
                            executor_name,
                            work_unit_label(work_unit),
                            dispatched_job.job_id or "n/a",
                            time.perf_counter() - started_at,
                        )
                    except Exception:
                        elapsed_s = time.perf_counter() - started_at
                        logger.exception(
                            "%s failed to dispatch %s after %.2fs.",
                            executor_name,
                            work_unit_label(work_unit),
                            elapsed_s,
                        )
                        runtime_event(
                            (
                                f"Dispatch failed for "
                                f"{task_label(work_unit.root, include_hash=False, include_arguments=True)} "
                                f"in {elapsed_s:.2f}s"
                            ),
                            style="bold red",
                        )
                        raise
                    progress_bar(1)

        dispatched_task_count = sum(len(wu.graph.nodes()) for wu in pending_work_units)
        completed_task_count = sum(len(wu.graph.nodes()) for wu in work_units) - dispatched_task_count
        summary = (
            f"Submitted {num_dispatch} {_plural(num_dispatch, 'job')} / "
            f"{dispatched_task_count} {_plural(dispatched_task_count, 'task')} "
            f"to {executor_name}"
        )
        if num_complete > 0:
            summary += (
                f" ({num_complete} {_plural(num_complete, 'job')} / "
                f"{completed_task_count} {_plural(completed_task_count, 'task')} already complete)"
            )
        runtime_event(summary, style="green bold")
        logger.info(
            "%s submitted %d work unit(s) (%d already complete).",
            executor_name,
            num_dispatch,
            num_complete,
        )

        # Keep graph topology and replace each WorkUnit node with its job handle.
        job_graph = cast("DependencyGraph[CompletedJob | JobT]", work_graph.copy())
        for i in job_graph.node_indices():
            job_graph[i] = jobs[work_graph[i]]

        if blocking:
            blocking_jobs = set(job_graph.nodes())
            logger.info("%s waiting for %d job(s) to reach terminal states.", executor_name, len(blocking_jobs))
            try:
                for job in blocking_jobs:
                    logger.debug(
                        "%s waiting on %s (job_id=%s).",
                        executor_name,
                        job.label,
                        job.job_id or "n/a",
                    )
                    job.wait()
                logger.info("%s observed all blocking jobs reach terminal states.", executor_name)

                failed_jobs = [job for job in blocking_jobs if job.state() == "failed"]
                if failed_jobs:
                    failed_labels = ", ".join(job.label for job in failed_jobs)
                    msg = f"{executor_name} observed {len(failed_jobs)} failed job(s): {failed_labels}"
                    logger.error(msg)
                    runtime_event(msg, style="bold red")
            finally:
                self.cleanup_snapshot(snapshot)

        return job_graph, snapshot

    def cleanup_snapshot(self, snapshot: Snapshot | None) -> None:
        """Clean up a snapshot created by :meth:`submit`.

        Args:
            snapshot: Snapshot to remove, or ``None`` (no-op).
        """
        if snapshot is not None:
            with runtime_activity("Cleaning up snapshot of the project environment", style="yellow"):
                snapshot.cleanup()
            logger.info("%s cleaned up snapshot.", self.__class__.__name__)
            runtime_event("Cleaned up snapshot of the project environment", style="green")

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

    def _make_local_snapshot(self, workspace: Workspace, snapshots_dir: Path | None = None) -> LocalSnapshot:
        """Create a fresh :class:`LocalSnapshot`.

        Args:
            workspace: Workspace used to derive the default snapshot parent
                directory when ``snapshots_dir`` is not provided.
            snapshots_dir: Optional override for the parent directory under
                which the per-snapshot directory is created. When ``None``,
                snapshots are placed under ``workspace.get_temp_dir() / "snapshots"``.
        """
        if snapshots_dir is None:
            snapshots_dir = workspace.get_temp_dir() / "snapshots"
        return LocalSnapshot(snapshots_dir=snapshots_dir.resolve())


class Job(ABC):
    """Abstract job handle returned by an executor backend.

    Job-log lifecycle is owned by the worker that produces the log
    (see :meth:`Workspace.streaming_job_log`), so this base class does
    not participate in log finalization -- subclasses just implement
    :meth:`state` directly.
    """

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

    @property
    def root(self) -> Task:
        """Return root task of the associated work unit."""
        return self.work_unit.root

    @property
    def resources(self) -> Resources:
        """Return aggregated resource requirements of the associated work unit."""
        return self.work_unit.resources

    @property
    def label(self) -> str:
        """Return compact human-readable label for this job."""
        return work_unit_label(self.work_unit)

    @abstractmethod
    def state(self) -> JobState:
        """Return the current backend job state."""

    @classmethod
    def bulk_state(cls, jobs: Sequence[Job]) -> dict[Job, JobState]:
        """Return the current state for each of ``jobs`` as a dict.

        The default implementation calls :meth:`state` once per job. Backends
        that can answer many jobs in a single backend call (e.g. one
        ``squeue`` invocation covering many SLURM job ids) should override
        this so the UI can poll many jobs without paying the per-job cost
        N times.

        Callers are expected to pass instances of ``cls`` (or to use
        :func:`bulk_job_states`, which does this grouping for them); a
        backend override may rely on that homogeneity.
        """
        return {job: job.state() for job in jobs}

    def wait(self, poll_s: float = 0.5) -> None:
        """Block until job reaches a terminal state.

        Args:
            poll_s: Polling interval in seconds.
        """
        while True:
            if self.state() in ("done", "failed"):
                return
            time.sleep(poll_s)


def bulk_job_states(jobs: Iterable[Job]) -> dict[Job, JobState]:
    """Return states for a heterogeneous set of jobs in as few calls as possible.

    Groups ``jobs`` by concrete class and dispatches one
    :meth:`Job.bulk_state` call per class. If a class's ``bulk_state``
    raises a known query error, every job in that group is reported as
    ``"unknown"`` rather than letting the failure propagate.
    """
    by_class: dict[type[Job], list[Job]] = {}
    for job in jobs:
        by_class.setdefault(type(job), []).append(job)
    result: dict[Job, JobState] = {}
    for klass, group in by_class.items():
        try:
            states = klass.bulk_state(group)
        except (FileNotFoundError, OSError, RuntimeError):
            for job in group:
                result[job] = "unknown"
            continue
        for job in group:
            state = states.get(job, "unknown")
            result[job] = state if state in _VALID_JOB_STATES else "unknown"
    return result


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

    @classmethod
    def bulk_state(cls, jobs: Sequence[Job]) -> dict[Job, JobState]:
        """All ``CompletedJob`` instances are unconditionally ``done``."""
        return dict.fromkeys(jobs, "done")
