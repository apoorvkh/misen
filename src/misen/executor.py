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
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, ClassVar, Generic, Literal, TypeAlias, TypeVar, cast

import msgspec

from misen.exceptions import JobFailedError, JobFailure, StatusQueryError, StorageError, SubmissionError
from misen.utils.hashing import TaskHash
from misen.utils.runtime_events import runtime_activity, runtime_event, runtime_progress, task_label, work_unit_label
from misen.utils.settings import Configurable
from misen.utils.work_unit import build_work_graph

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

    from misen.task_metadata import Resources
    from misen.tasks import Task
    from misen.utils.graph import DependencyGraph
    from misen.utils.snapshot import ProjectSnapshot
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = ["Executor", "Job", "JobState", "bulk_job_states", "raise_for_failed_jobs"]

ExecutorType: TypeAlias = Literal["local", "in_process", "slurm", "skypilot"]
JobState: TypeAlias = Literal["pending", "running", "done", "failed", "unknown"]
_VALID_JOB_STATES: frozenset[JobState] = frozenset({"pending", "running", "done", "failed", "unknown"})
JobT = TypeVar("JobT", bound="Job")
logger = logging.getLogger(__name__)


class _JobRecord(msgspec.Struct):
    job_id: str
    native_id: str | int
    submission_id: str = ""
    deadline_minutes: int = 0
    version: Literal[1] = 1
    request_id: str | None = None
    native_name: str | None = None


class Executor(Configurable, Generic[JobT]):
    """Abstract execution backend interface.

    Shared submission logic here handles dependency-aware graph traversal,
    completed-work short circuiting, and snapshot creation; subclasses
    provide dispatch behavior.
    """

    _config_key: ClassVar[str] = "executor"
    _config_default_type: ClassVar[str] = "misen.executors.local:LocalExecutor"
    _config_aliases: ClassVar[dict[ExecutorType, str]] = {
        "local": "misen.executors.local:LocalExecutor",
        "in_process": "misen.executors.in_process:InProcessExecutor",
        "slurm": "misen.executors.slurm:SlurmExecutor",
        "skypilot": "misen.executors.skypilot:SkyPilotExecutor",
    }

    snapshot: bool = True
    env_store_dir: str | None = None
    prewarm_envs: bool = False
    _job_class: ClassVar[type[Job] | None] = None

    def session(self) -> AbstractContextManager[None]:
        """Keep backend services alive across submission and job observation.

        Most executors require no scoped resources. Managed SkyPilot API
        servers use this boundary for nonblocking Python submissions; CLI
        runs and blocking submissions establish it automatically.
        """
        return nullcontext()

    def _run_defaults_to_blocking(self) -> bool:
        """Whether Experiment.run must keep an attached coordinator alive."""
        return False

    def submit(
        self,
        tasks: set[Task],
        workspace: Workspace,
        *,
        blocking: bool = False,
    ) -> DependencyGraph[CompletedJob | JobT]:
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
            Dependency graph of job handles. Submission artifacts (the
            published snapshot, per-job payloads, env-file copies) are
            retained in the workspace — payloads must outlive scheduler
            requeues, so nothing is deleted when a submit call ends.

        Raises:
            MisenError: If configuration, snapshots, storage, submission, or
                status handling fails at a Misen-owned boundary.
            JobFailedError: If a blocking submission contains failed jobs.
            ValueError: If task resource requests cannot form a valid work
                graph, such as an invalid Dask topology.
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

        snapshot: ProjectSnapshot | None = None
        if pending_work_units:
            from misen.utils.snapshot import ProjectSnapshot, _detect_pixi_wrap

            self._validate_submission(work_graph=work_graph, pending_work_units=pending_work_units, workspace=workspace)
            logger.info("%s creating snapshot for %d pending work unit(s).", executor_name, num_dispatch)
            started_at = time.perf_counter()
            try:
                with runtime_activity("Creating a snapshot of the project environment", style="yellow"):
                    if self.snapshot:
                        snapshot = ProjectSnapshot(
                            workspace=workspace,
                            env_store_dir=self.env_store_dir,
                            prewarm=self.prewarm_envs,
                        )
                    else:
                        _detect_pixi_wrap()  # fail fast on a misconfigured pixi.lock
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
                self._dispatch_work_graph(
                    pending_work_units=pending_work_units,
                    jobs=jobs,
                    workspace=workspace,
                    snapshot=snapshot,
                    progress=progress_bar,
                )

        task_counts = {work_unit: len(work_unit.graph.nodes()) for work_unit in work_units}
        dispatched_task_count = sum(task_counts[work_unit] for work_unit in pending_work_units)
        completed_task_count = sum(task_counts.values()) - dispatched_task_count
        summary = f"Submitted {num_dispatch} job(s) / {dispatched_task_count} task(s) to {executor_name}"
        if num_complete > 0:
            summary += f" ({num_complete} job(s) / {completed_task_count} task(s) already complete)"
        runtime_event(summary, style="green bold")
        logger.info(
            "%s submitted %d work unit(s) (%d already complete).",
            executor_name,
            num_dispatch,
            num_complete,
        )

        # Keep graph topology and replace each WorkUnit node with its job handle in place.
        job_graph = cast("DependencyGraph[CompletedJob | JobT]", work_graph)
        for i in job_graph.node_indices():
            job_graph[i] = jobs[cast("WorkUnit", job_graph[i])]

        if blocking:
            blocking_jobs = list(job_graph.nodes())
            logger.info("%s waiting for %d job(s) to reach terminal states.", executor_name, len(blocking_jobs))
            # One batched query per tick (e.g. a single squeue+sacct pair
            # for all SLURM jobs) instead of per-job polling.
            states = bulk_job_states(blocking_jobs)
            while any(state not in ("done", "failed") for state in states.values()):
                time.sleep(0.5)
                states = bulk_job_states(blocking_jobs)
            logger.info("%s observed all blocking jobs reach terminal states.", executor_name)

            raise_for_failed_jobs(states, context=executor_name)

        return job_graph

    def _validate_submission(
        self,
        *,
        work_graph: DependencyGraph[WorkUnit],
        pending_work_units: Sequence[WorkUnit],
        workspace: Workspace,
    ) -> None:
        """Validate backend-specific submission compatibility before snapshotting.

        Workflow and remote backends can reject unsupported graph shapes or
        workspace transports here, before project staging performs any work.
        Per-work-unit resource validation may still happen during dispatch.

        The default accepts every submission. Backends should override this
        only when they have additional preflight constraints.
        """
        del work_graph, pending_work_units, workspace

    def _dispatch_work_graph(
        self,
        *,
        pending_work_units: Sequence[WorkUnit],
        jobs: dict[WorkUnit, CompletedJob | JobT],
        workspace: Workspace,
        snapshot: ProjectSnapshot | None,
        progress: Callable[[int], None],
    ) -> None:
        """Dispatch pending work units, preserving the default eager behavior.

        Backends with a native workflow primitive may override this hook to
        submit the pending graph atomically. The default implementation calls
        :meth:`_dispatch` in dependency order and retains partial-submission
        diagnostics for schedulers such as SLURM.
        """
        executor_name = self.__class__.__name__
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
                dispatched_job = self._dispatch_or_reattach(
                    work_unit, cast("set[JobT]", dependencies), workspace, snapshot
                )
                jobs[work_unit] = dispatched_job
                logger.debug(
                    "%s dispatched %s (job_id=%s) in %.2fs.",
                    executor_name,
                    work_unit_label(work_unit),
                    dispatched_job.job_id or "n/a",
                    time.perf_counter() - started_at,
                )
            except Exception as exc:
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
                submitted = tuple(job for job in jobs.values() if not isinstance(job, CompletedJob))
                if submitted and isinstance(exc, SubmissionError):
                    msg = (
                        f"Could not submit {work_unit_label(work_unit)} after "
                        f"{len(submitted)} earlier job(s) were accepted: {exc}"
                    )
                    raise SubmissionError(msg, submitted_jobs=(*submitted, *exc.submitted_jobs)) from exc
                if submitted:
                    labels = ", ".join(job.label for job in submitted)
                    exc.add_note(f"Already submitted jobs: {labels}.")
                raise
            progress(1)

    def _dispatch_or_reattach(
        self,
        work_unit: WorkUnit,
        dependencies: set[JobT],
        workspace: Workspace,
        snapshot: ProjectSnapshot | None,
    ) -> JobT:
        """Reuse a live durable job, or atomically submit and record one."""
        if self._job_class is None:
            return self._dispatch(work_unit, dependencies, workspace, snapshot)
        snapshot_key = getattr(snapshot, "snapshot_key", None)
        key = TaskHash.from_object(
            (self._job_key_identity(), work_unit.root.task_hash(), work_unit.resources, snapshot_key)
        ).b32()
        with workspace.lock("job", key).context():
            job = self._restore_job(work_unit, workspace, key)
            if job is not None:
                previous_record = job._record()  # noqa: SLF001
                try:
                    restored_state = job.state()
                finally:
                    # A backend may resolve a provisional native identity
                    # before a later part of its status query fails. Preserve
                    # only an actual transition while the record lock is held.
                    if job._record() != previous_record:  # noqa: SLF001
                        self._record_job(job, workspace, key)
            else:
                restored_state = None
            if job is not None and restored_state not in ("done", "failed"):
                job._bind_record(workspace, key)  # noqa: SLF001
                logger.info("Reattached %s (job_id=%s).", work_unit_label(work_unit), job.job_id)
                return job
            job = self._dispatch(work_unit, dependencies, workspace, snapshot)
            try:
                self._record_job(job, workspace, key)
            except Exception as exc:
                msg = f"Accepted {job.label}, but could not persist its durable job record: {exc}"
                raise SubmissionError(msg, submitted_jobs=(job,)) from exc
            job._bind_record(workspace, key)  # noqa: SLF001
            return job

    def _job_key_identity(self) -> object:
        """Return the executor identity used by durable reattachment keys."""
        return self

    def _restore_job(self, work_unit: WorkUnit, workspace: Workspace, key: str) -> JobT | None:
        """Restore a backend job record, or return ``None`` when absent."""
        try:
            record = msgspec.json.decode(workspace.read_job_file("jobs", f"{key}.json"), type=_JobRecord)
        except FileNotFoundError:
            return None
        except msgspec.DecodeError as exc:
            msg = f"Invalid durable job record {key}: {exc}"
            raise StorageError(msg) from exc
        job_class = cast("type[Job]", self._job_class)
        return cast("JobT", job_class._from_record(work_unit, workspace, record))  # noqa: SLF001

    def _record_job(self, job: JobT, workspace: Workspace, key: str) -> None:
        """Persist the backend fields required to restore ``job``."""
        workspace.put_job_file("jobs", f"{key}.json", msgspec.json.encode(job._record()))  # noqa: SLF001

    @abstractmethod
    def _dispatch(
        self, work_unit: WorkUnit, dependencies: set[JobT], workspace: Workspace, snapshot: ProjectSnapshot | None
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


class Job(ABC):
    """Abstract job handle returned by an executor backend.

    Subclasses implement backend state queries; this base class supplies
    bounded polling, failure diagnostics, and terminal log finalization.
    """

    __slots__ = (
        "_failure_reason",
        "_log_finalized",
        "_record_key",
        "_record_workspace",
        "_unknown_since",
        "job_id",
        "log_path",
        "work_unit",
    )

    unknown_state_timeout_s: ClassVar[float] = 60.0

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
        self._failure_reason: str | None = None
        self._log_finalized = False
        self._record_key: str | None = None
        self._record_workspace: Workspace | None = None
        self._unknown_since: float | None = None

    def _bind_record(self, workspace: Workspace, key: str) -> None:
        """Remember where this durable job handle is persisted."""
        self._record_workspace = workspace
        self._record_key = key

    def _refresh_record(self) -> None:
        """Persist a native-identity transition without clobbering a replacement."""
        if self._record_workspace is None or self._record_key is None:
            return
        workspace = self._record_workspace
        name = f"{self._record_key}.json"
        with workspace.lock("job", self._record_key).context():
            try:
                current = msgspec.json.decode(workspace.read_job_file("jobs", name), type=_JobRecord)
            except FileNotFoundError as exc:
                msg = f"Durable job record {self._record_key!r} disappeared during an identity transition."
                raise StorageError(msg) from exc
            except msgspec.DecodeError as exc:
                msg = f"Invalid durable job record {self._record_key}: {exc}"
                raise StorageError(msg) from exc
            if current.job_id != self.job_id:
                logger.info(
                    "Skipped stale durable-record refresh for job_id=%s; current job_id=%s.",
                    self.job_id,
                    current.job_id,
                )
                return
            workspace.put_job_file("jobs", name, msgspec.json.encode(self._record()))

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

    @property
    def failure(self) -> JobFailure:
        """Return immutable diagnostic facts for this job."""
        return JobFailure(
            label=self.label,
            job_id=self.job_id,
            log_path=str(self.log_path) if self.log_path is not None else None,
            reason=self._failure_reason,
        )

    def _record_failure(self, reason: str) -> None:
        """Record a concise failure reason while retaining earlier diagnostics."""
        if self._failure_reason is None:
            self._failure_reason = reason
        elif reason not in self._failure_reason:
            self._failure_reason = f"{self._failure_reason} {reason}"

    def _finalize_log(self, workspace: Workspace, *, failed: bool) -> None:
        """Publish a terminal log without replacing an existing job failure."""
        if self._log_finalized or self.log_path is None:
            return
        try:
            workspace.finalize_job_log(self.log_path)
        except (OSError, StorageError) as exc:
            if not failed:
                if isinstance(exc, StorageError):
                    raise
                msg = f"Could not finalize terminal job log {self.log_path}: {exc}"
                raise StorageError(msg) from exc
            self._record_failure(f"Additionally, finalizing log {self.log_path} failed: {type(exc).__name__}: {exc}")
            logger.exception("Could not finalize the failed job log %s.", self.log_path)
        else:
            self._log_finalized = True

    def _normalize_state(self, state: object, *, queried_at: float) -> JobState:
        """Normalize one backend state and bound consecutive unknown results."""
        resolved = state if isinstance(state, str) and state in _VALID_JOB_STATES else "unknown"
        if resolved != "unknown":
            self._unknown_since = None
            return resolved
        if self._unknown_since is None:
            self._unknown_since = queried_at
        elif queried_at - self._unknown_since >= self.unknown_state_timeout_s:
            job_id = f" (job_id={self.job_id})" if self.job_id is not None else ""
            msg = f"Could not determine status for {self.label}{job_id} for {self.unknown_state_timeout_s:g}s."
            raise StatusQueryError(msg)
        return "unknown"

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

        Raises:
            StatusQueryError: If status cannot be determined within the retry
                budget.
        """
        while bulk_job_states([self])[self] not in ("done", "failed"):
            time.sleep(poll_s)

    def raise_for_status(self, state: JobState | None = None) -> None:
        """Raise if this job has failed.

        Args:
            state: Previously queried state, avoiding another backend call.

        Raises:
            JobFailedError: If the resolved state is ``"failed"``.
            StatusQueryError: If no state was supplied and querying it exceeds
                the retry budget.
        """
        resolved_state = bulk_job_states([self])[self] if state is None else state
        raise_for_failed_jobs({self: resolved_state})

    def cancel(self) -> None:
        """Cancel this job when its backend supports native cancellation."""
        msg = f"{type(self).__name__} does not support cancellation."
        raise NotImplementedError(msg)

    @classmethod
    def _from_record(cls, work_unit: WorkUnit, workspace: Workspace, record: _JobRecord) -> Job:
        del work_unit, workspace, record
        msg = f"{cls.__name__} does not support durable reattachment."
        raise NotImplementedError(msg)

    def _record(self) -> _JobRecord:
        msg = f"{type(self).__name__} does not support durable reattachment."
        raise NotImplementedError(msg)


def bulk_job_states(jobs: Iterable[Job]) -> dict[Job, JobState]:
    """Return states for a heterogeneous set of jobs in as few calls as possible.

    Groups ``jobs`` by concrete class and dispatches one
    :meth:`Job.bulk_state` call per class. Known backend query failures are
    translated to :class:`StatusQueryError` so callers do not poll forever.

    Raises:
        StatusQueryError: If a backend keeps failing or returning unknown
            states beyond its retry budget.
    """
    by_class: dict[type[Job], list[Job]] = {}
    for job in jobs:
        by_class.setdefault(type(job), []).append(job)
    result: dict[Job, JobState] = {}
    for klass, group in by_class.items():
        query_error: BaseException | None = None
        try:
            states = klass.bulk_state(group)
        except (OSError, RuntimeError, StatusQueryError) as exc:
            if isinstance(exc, StatusQueryError) and not exc.retryable:
                raise
            states = {}
            query_error = exc
        queried_at = time.monotonic()
        try:
            result.update(
                {
                    job: job._normalize_state(states.get(job, "unknown"), queried_at=queried_at)  # noqa: SLF001
                    for job in group
                }
            )
        except StatusQueryError:
            if query_error is not None:
                msg = f"Could not query {klass.__name__} job states: {query_error}"
                raise StatusQueryError(msg) from query_error
            raise
    return result


def raise_for_failed_jobs(states: Mapping[Job, JobState], *, context: str | None = None) -> None:
    """Raise one structured error when any queried job has failed.

    Args:
        states: Previously queried state for each job.
        context: Optional executor or operation name for the summary.

    Raises:
        JobFailedError: If one or more states are ``"failed"``.
    """
    failures = tuple(job.failure for job, state in states.items() if state == "failed")
    if not failures:
        return

    def describe(failure: JobFailure) -> str:
        details: list[str] = []
        if failure.job_id is not None:
            details.append(f"job_id={failure.job_id}")
        if failure.reason is not None:
            details.append(failure.reason)
        if failure.log_path is not None:
            details.append(f"log={failure.log_path}")
        return f"{failure.label} ({', '.join(details)})" if details else failure.label

    prefix = f"{context} observed " if context is not None else ""
    noun = "job" if len(failures) == 1 else "jobs"
    msg = f"{prefix}{len(failures)} failed {noun}: {'; '.join(map(describe, failures))}"
    raise JobFailedError(msg, failures=failures)


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
