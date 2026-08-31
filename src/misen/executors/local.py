"""Local subprocess-based executor implementation."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import logging
import os
import signal
import sys
import threading
from bisect import insort
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from processkit import Command, ProcessError, ProcessResult, SupervisionSession, Supervisor

from misen.exceptions import ExecutionError, StorageError, SubmissionError
from misen.executor import Executor, Job
from misen.task_metadata import AcceleratorType  # noqa: TC001  # needed by msgspec during config decoding
from misen.utils.resource_env import resource_environment
from misen.utils.runtime_events import (
    runtime_job_done,
    runtime_job_failed,
    runtime_job_pending,
    runtime_job_running,
    task_label,
    work_unit_label,
)
from misen.utils.snapshot import prepare_live_job

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType

    from misen.task_metadata import Resources
    from misen.utils.snapshot import ProjectSnapshot
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = ("LocalExecutor", "LocalJob")

_JobState = Literal["pending", "running", "done", "failed"]
logger = logging.getLogger(__name__)
_TIMEOUT_KILL_GRACE_S = 5.0
_SHUTDOWN_TERM_GRACE_S = 2.0
_SIGTERM_CLEANUP_TIMEOUT_S = 5.0
_USER_CANCEL_REASON = "Canceled by user."


def _inherited_cpu_indices() -> list[int]:
    """Return the CPU pool this process may safely pass to its children."""
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            cpu_indices = sorted(sched_getaffinity(0))
        except (OSError, ValueError):
            pass
        else:
            if cpu_indices:
                return cpu_indices
    return list(range(os.cpu_count() or 1))


class LocalJob(Job):
    """Job handle backed by a local subprocess."""

    __slots__ = (
        "_cached_state",
        "_lock",
        "_process",
        "_scheduler",
        "assigned_accelerator_indices",
        "assigned_cpu_indices",
        "dependencies",
        "snapshot",
        "workspace",
    )

    def __init__(
        self,
        work_unit: WorkUnit,
        dependencies: set[LocalJob],
        snapshot: ProjectSnapshot | None,
        workspace: Workspace,
    ) -> None:
        """Initialize a local job."""
        super().__init__(work_unit=work_unit, job_id=None, log_path=None)
        self.dependencies = set(dependencies)
        self.snapshot = snapshot
        self.workspace = workspace
        self.assigned_cpu_indices: list[int] = []
        self.assigned_accelerator_indices: list[int] = []
        self._process: SupervisionSession | None = None
        self._scheduler: _LocalScheduler | None = None
        self._cached_state: _JobState = "pending"
        self._lock = threading.Lock()

    def state(self) -> _JobState:
        """Return the current process-backed job state."""
        with self._lock:
            if self._cached_state in {"done", "failed"}:
                self._finalize_job_log_locked()
                return self._cached_state
            if self._process is None:
                return self._cached_state

            process = self._process
            try:
                if process.status.is_active:
                    return "running"
                self._process = None
                result = process.wait().final_result
            except (ProcessError, RuntimeError) as exc:
                self._process = None
                self._cached_state = "failed"
                self._record_failure(f"Process supervision failed: {type(exc).__name__}: {exc}")
                self._finalize_job_log_locked()
                return self._cached_state
            return self._finish_locked(result)

    def _finish_locked(self, result: ProcessResult) -> _JobState:
        """Apply one terminal process result and publish the completed log."""
        self._process = None
        if self.failure.reason is None:
            if result.timed_out:
                self._record_failure(f"Exceeded the requested {self.resources['time']} minute time limit.")
            elif not result.is_success:
                if result.code is not None:
                    self._record_failure(f"Process exited with code {result.code}.")
                elif result.signal is not None:
                    self._record_failure(f"Process terminated by signal {result.signal}.")
                else:
                    self._record_failure("Process failed without an exit code or signal.")
        self._cached_state = "done" if result.is_success and self.failure.reason is None else "failed"
        self._finalize_job_log_locked()
        logger.info(
            "Local job %s for %s exited (code=%s, signal=%s, timed_out=%s) -> state=%s.",
            self.job_id or "n/a",
            self.label,
            result.code,
            result.signal,
            result.timed_out,
            self._cached_state,
        )
        return self._cached_state

    def _finalize_job_log_locked(self) -> None:
        """Publish the terminal log once the worker process has exited."""
        if self._cached_state not in {"done", "failed"}:
            return
        self._finalize_log(self.workspace, failed=self._cached_state == "failed")

    def set_process(
        self,
        process: SupervisionSession,
        *,
        cpu_indices: list[int],
        accelerator_indices: list[int],
    ) -> int | None:
        """Attach process supervision and mark the job running."""
        pid = process.status.pid
        with self._lock:
            self._process = process
            self.assigned_cpu_indices = list(cpu_indices)
            self.assigned_accelerator_indices = list(accelerator_indices)
            self._cached_state = "running"
            logger.info(
                "Local job %s for %s is running (pid=%s).",
                self.job_id or "n/a",
                self.label,
                pid or "pending",
            )
            return pid

    def mark_failed(self, reason: str | None = None) -> None:
        """Mark a pending/running job failed."""
        with self._lock:
            if reason is not None:
                self._record_failure(reason)
            self._cached_state = "failed"
            logger.error("Local job %s for %s marked failed.", self.job_id or "n/a", self.label)

    def cancel(self) -> None:
        """Cancel this job without allowing queued work to launch later."""
        state = self.state()
        if state in {"done", "failed"}:
            return
        with self._lock:
            scheduler = self._scheduler
        if scheduler is not None:
            scheduler.cancel(self)
        elif state == "running":
            self.stop(grace_seconds=_SHUTDOWN_TERM_GRACE_S, reason=_USER_CANCEL_REASON)
        else:
            self.mark_failed(reason=_USER_CANCEL_REASON)

    def _associate_scheduler(self, scheduler: _LocalScheduler) -> None:
        """Attach the scheduler that owns this job's queue bookkeeping."""
        with self._lock:
            self._scheduler = scheduler

    def stop(self, *, grace_seconds: float, reason: str | None = None) -> bool:
        """Stop and reap the supervised process tree within a grace window."""
        process = self._detach_process(reason)
        if process is None:
            return True
        try:
            result = process.stop(grace_seconds).final_result
        except (ProcessError, RuntimeError) as exc:
            return self._record_stop_failure(exc)
        return self._record_stopped(result)

    async def astop(self, *, grace_seconds: float, reason: str | None = None) -> bool:
        """Asynchronously stop and reap the supervised process tree."""
        process = self._detach_process(reason)
        if process is None:
            return True
        try:
            result = (await process.astop(grace_seconds)).final_result
        except (ProcessError, RuntimeError) as exc:
            return self._record_stop_failure(exc)
        return self._record_stopped(result)

    def _detach_process(self, reason: str | None) -> SupervisionSession | None:
        """Claim the live session for one terminal operation."""
        with self._lock:
            if self._process is None:
                self._finalize_job_log_locked()
                return None
            if reason is not None:
                self._record_failure(reason)
            process = self._process
            self._process = None
            return process

    def _record_stop_failure(self, exc: Exception) -> bool:
        """Record a failed processkit terminal operation."""
        with self._lock:
            self._record_failure(f"Stopping process supervision failed: {type(exc).__name__}: {exc}")
            self._cached_state = "failed"
            self._finalize_job_log_locked()
            logger.error("Could not stop local job %s for %s: %s", self.job_id or "n/a", self.label, exc)
            return False

    def _record_stopped(self, result: ProcessResult) -> bool:
        """Apply the result of a successful processkit stop."""
        with self._lock:
            self._finish_locked(result)
            return True

    def force_kill(self, *, reason: str | None = None) -> bool:
        """Immediately stop and reap the supervised process tree."""
        return self.stop(grace_seconds=0, reason=reason)


class LocalExecutor(Executor[LocalJob]):
    """Executor that runs work units as local subprocesses.

    Snapshots are content-addressed project state published to the
    workspace; environments materialize from them into the env store at
    ``env_store_dir`` (default ``/tmp/misen-env-store-<user>``). With
    ``prewarm_envs`` (the default here) they are built once at submission —
    concurrent jobs share them and env failures surface before any job
    starts; with ``prewarm_envs=False`` the first job builds them at
    startup instead.

    Resource requests are scheduling controls. ``max_memory`` is an aggregate
    admission budget for declared job memory, not a per-job OS-enforced limit.
    Accelerator assignments use cooperative runtime visibility variables, not
    a device-access security boundary.
    """

    max_memory: int | Literal["all"] = "all"
    num_cpus: int | Literal["all"] = "all"
    cpu_indices: list[int] | None = None
    accelerators: int = 0
    accelerator_indices: list[int] | None = None
    accelerator_type: AcceleratorType = "cuda"
    prewarm_envs: bool = True
    enforce_time_limits: bool = False
    _config_validation_errors: ClassVar[tuple[type[Exception], ...]] = (ValueError,)

    def __post_init__(self) -> None:
        """Infer scheduling capacity and initialize the scheduler."""
        if self.max_memory == "all":
            try:
                self.max_memory = max(
                    1,
                    os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024**3),
                )
            except (ValueError, OSError, AttributeError):
                self.max_memory = 1
        elif isinstance(self.max_memory, bool) or not isinstance(self.max_memory, int) or self.max_memory < 1:
            msg = "max_memory must be 'all' or a positive integer number of GiB."
            raise ValueError(msg)

        if self.num_cpus != "all" and self.cpu_indices is not None:
            msg = "num_cpus and cpu_indices should not both be passed to LocalExecutor."
            raise ValueError(msg)
        inherited_cpu_indices = _inherited_cpu_indices()
        if self.cpu_indices is None:
            if self.num_cpus == "all":
                cpu_indices = inherited_cpu_indices
            elif isinstance(self.num_cpus, bool) or not isinstance(self.num_cpus, int) or self.num_cpus < 1:
                msg = "num_cpus must be 'all' or a positive integer."
                raise ValueError(msg)
            elif self.num_cpus > len(inherited_cpu_indices):
                msg = (
                    f"num_cpus={self.num_cpus} exceeds the {len(inherited_cpu_indices)} CPU(s) "
                    "available to this process."
                )
                raise ValueError(msg)
            else:
                cpu_indices = inherited_cpu_indices[: self.num_cpus]
        else:
            if not self.cpu_indices or any(
                isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in self.cpu_indices
            ):
                msg = "cpu_indices must contain nonnegative integer CPU indices."
                raise ValueError(msg)
            cpu_indices = sorted(set(self.cpu_indices))
            unavailable_cpu_indices = sorted(set(cpu_indices).difference(inherited_cpu_indices))
            if unavailable_cpu_indices:
                msg = f"cpu_indices contains CPUs unavailable to this process: {unavailable_cpu_indices}."
                raise ValueError(msg)

        if self.accelerator_type not in ("cuda", "rocm", "xpu", "mps", "tpu"):
            msg = f"Unsupported accelerator type: {self.accelerator_type!r}."
            raise ValueError(msg)
        if isinstance(self.accelerators, bool) or not isinstance(self.accelerators, int) or self.accelerators < 0:
            msg = "accelerators must be a nonnegative integer."
            raise ValueError(msg)
        if self.accelerators and self.accelerator_indices is not None:
            msg = "accelerators and accelerator_indices should not both be passed to LocalExecutor."
            raise ValueError(msg)
        if self.accelerator_indices is None:
            accelerator_indices = list(range(self.accelerators)) if self.accelerator_type != "tpu" else []
        else:
            if self.accelerator_type == "tpu":
                msg = "LocalExecutor does not support accelerator_indices for TPUs."
                raise ValueError(msg)
            if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in self.accelerator_indices):
                msg = "accelerator_indices must contain nonnegative integer indices."
                raise ValueError(msg)
            accelerator_indices = sorted(set(self.accelerator_indices))
        accelerator_count = self.accelerators or len(accelerator_indices)
        if self.accelerator_type == "mps" and accelerator_indices not in ([], [0]):
            msg = "MPS exposes a single GPU at index 0."
            raise ValueError(msg)

        self._resource_budget = _ResourceBudget(
            memory=self.max_memory,
            cpus=len(cpu_indices),
            accelerators=accelerator_count,
            accelerator_type=self.accelerator_type,
        )
        self._scheduler = _LocalScheduler(
            available_budget=self._resource_budget,
            available_cpu_indices=cpu_indices,
            available_accelerator_indices=accelerator_indices,
            enforce_time_limits=self.enforce_time_limits,
        )
        logger.info(
            "Initialized LocalExecutor budget: memory=%sGiB cpus=%d accelerators=%d type=%s.",
            self._resource_budget.memory,
            self._resource_budget.cpus,
            self._resource_budget.accelerators,
            self._resource_budget.accelerator_type,
        )

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[LocalJob],
        workspace: Workspace,
        snapshot: ProjectSnapshot | None,
    ) -> LocalJob:
        """Queue a work unit in the local scheduler."""
        resources = work_unit.resources
        if resources["nodes"] != 1:
            msg = "LocalExecutor supports only single-node tasks."
            raise SubmissionError(msg)
        if resources["accelerator_memory"] is not None:
            msg = "LocalExecutor cannot verify accelerator memory."
            raise SubmissionError(msg)
        if not self._resource_budget.fits(resources):
            msg = (
                "Requested resources exceed LocalExecutor capacity: "
                f"requested cpus={resources['cpus']}, memory={resources['memory']}, "
                f"accelerators={resources['accelerators']} (type={resources['accelerator_type']}); "
                f"capacity cpus={self._resource_budget.cpus}, memory={self._resource_budget.memory}, "
                f"accelerators={self._resource_budget.accelerators} "
                f"(type={self._resource_budget.accelerator_type})."
            )
            raise SubmissionError(msg)

        job = LocalJob(
            work_unit=work_unit,
            dependencies=dependencies,
            snapshot=snapshot,
            workspace=workspace,
        )
        logger.debug(
            "Queued local work unit %s with resources=%s and %d dependency job(s).",
            work_unit_label(work_unit),
            resources,
            len(dependencies),
        )
        runtime_job_pending(
            id(job),
            label=task_label(work_unit.root, include_hash=False, include_arguments=True),
        )
        self._scheduler.submit(job)
        return job


@dataclass(frozen=True, slots=True)
class _ResourceBudget:
    memory: int
    cpus: int
    accelerators: int
    accelerator_type: AcceleratorType

    def fits(self, resources: Resources) -> bool:
        if resources["cpus"] > self.cpus or resources["memory"] > self.memory:
            return False
        if resources["accelerators"] == 0:
            return True
        count_fits = (
            resources["accelerators"] == self.accelerators
            if resources["accelerator_type"] == "tpu"
            else resources["accelerators"] <= self.accelerators
        )
        return resources["accelerator_type"] == self.accelerator_type and count_fits

    def add(self, resources: Resources) -> _ResourceBudget:
        return self._adjust(resources, 1)

    def subtract(self, resources: Resources) -> _ResourceBudget:
        return self._adjust(resources, -1)

    def _adjust(self, resources: Resources, sign: Literal[-1, 1]) -> _ResourceBudget:
        return _ResourceBudget(
            memory=self.memory + resources["memory"] * sign,
            cpus=self.cpus + resources["cpus"] * sign,
            accelerators=self.accelerators + resources["accelerators"] * sign,
            accelerator_type=self.accelerator_type,
        )


class _LocalScheduler:
    """Event-driven local scheduler with bounded resource backfilling.

    Dependency edges are visited only when jobs finish. Newly ready jobs may
    backfill, while resource-blocked jobs retry in FIFO order.
    """

    __slots__ = (
        "_blocked",
        "_condition",
        "_dependents",
        "_fatal_error",
        "_ready",
        "_running",
        "_shutting_down",
        "_thread",
        "_waiting",
        "available_accelerator_indices",
        "available_budget",
        "available_cpu_indices",
        "enforce_time_limits",
    )
    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        available_budget: _ResourceBudget,
        available_cpu_indices: list[int],
        available_accelerator_indices: list[int],
        enforce_time_limits: bool = False,
    ) -> None:
        self.available_budget = available_budget
        self.available_cpu_indices = list(available_cpu_indices)
        self.available_accelerator_indices = list(available_accelerator_indices)
        self.enforce_time_limits = enforce_time_limits
        self._blocked: deque[LocalJob] = deque()
        self._dependents: dict[LocalJob, list[LocalJob]] = {}
        self._ready: deque[LocalJob] = deque()
        self._running: dict[LocalJob, None] = {}
        self._waiting: dict[LocalJob, int] = {}
        self._fatal_error: Exception | None = None
        self._shutting_down = False
        self._condition = threading.Condition()
        self._thread = threading.Thread(name="misen-local-scheduler", target=self._run, daemon=True)
        self._thread.start()
        atexit.register(self._terminate_running_jobs)
        _install_sigterm_handler(self._terminate_running_jobs)
        self._logger.info("Started LocalScheduler background thread.")

    def submit(self, job: LocalJob) -> None:
        """Queue a job for scheduling."""
        with self._condition:
            if self._shutting_down:
                msg = "Local scheduler is shutting down."
                raise ExecutionError(msg)
            if self._fatal_error is not None:
                msg = f"Local scheduler is unavailable after {type(self._fatal_error).__name__}: {self._fatal_error}"
                raise ExecutionError(msg) from self._fatal_error
            job._associate_scheduler(self)  # noqa: SLF001
            dependencies = [(dependency, dependency.state()) for dependency in job.dependencies]
            waiting = [dependency for dependency, state in dependencies if state != "done"]
            if any(state == "failed" for _, state in dependencies):
                self._mark_failed_locked(job, reason="A dependency failed.")
            elif waiting:
                self._waiting[job] = len(waiting)
                for dependency in waiting:
                    self._dependents.setdefault(dependency, []).append(job)
            else:
                self._ready.append(job)
            self._logger.debug(
                "Queued job for %s (pending=%d, running=%d).",
                job.work_unit,
                len(self._ready) + len(self._blocked) + len(self._waiting),
                len(self._running),
            )
            self._condition.notify_all()

    def cancel(self, job: LocalJob) -> None:
        """Cancel one tracked job and reconcile its scheduler bookkeeping."""
        with self._condition:
            if job.state() in {"done", "failed"}:
                return
            if job in self._running:
                running = True
            elif self._discard_pending_locked(job):
                self._mark_failed_locked(job, reason=_USER_CANCEL_REASON)
                self._condition.notify_all()
                return
            else:
                return

        try:
            job.stop(grace_seconds=_SHUTDOWN_TERM_GRACE_S, reason=_USER_CANCEL_REASON)
        finally:
            # Stopping a process may wait through the grace period, so it must
            # happen without holding the scheduler condition. Reconcile after
            # re-entering the critical section; the scheduler thread may have
            # collected the job already in the meantime.
            with self._condition:
                if running and job in self._running:
                    finished = self._collect_finished_locked()
                    if finished and not self._shutting_down:
                        self._retry_blocked_locked()
                    if not self._shutting_down:
                        self._start_ready_jobs_locked()
                self._condition.notify_all()

    def _discard_pending_locked(self, job: LocalJob) -> bool:
        """Remove a pending job from every queue and reverse dependency edge."""
        removed = False
        for queue in (self._ready, self._blocked):
            while True:
                try:
                    queue.remove(job)
                except ValueError:
                    break
                removed = True
        if job in self._waiting:
            del self._waiting[job]
            removed = True

        # A waiting job is indexed under each unfinished dependency. Remove
        # those reverse edges now so a later dependency completion cannot
        # requeue the canceled job. Keep the job's own dependent list intact;
        # _mark_failed_locked consumes it to fail descendants transitively.
        for dependency, dependents in tuple(self._dependents.items()):
            if dependency is job or job not in dependents:
                continue
            self._dependents[dependency] = [dependent for dependent in dependents if dependent is not job]
            if not self._dependents[dependency]:
                del self._dependents[dependency]
        return removed

    def _run(self) -> None:
        while True:
            try:
                with self._condition:
                    if self._shutting_down:
                        return
                    self._tick_locked()
                    self._condition.wait(timeout=0.1 if self._running else None)
            except Exception as exc:
                self._logger.exception("LocalScheduler stopped after an unexpected error.")
                with self._condition:
                    self._fatal_error = exc
                    self._fail_all_locked(f"Local scheduler failed: {type(exc).__name__}: {exc}")
                    self._condition.notify_all()
                return

    def _tick_locked(self) -> None:
        if self._shutting_down:
            return
        if self._collect_finished_locked():
            self._retry_blocked_locked()
        self._start_ready_jobs_locked()

    def _collect_finished_locked(self) -> bool:
        finished_any = False
        for job in list(self._running):
            try:
                state = job.state()
            except StorageError as exc:
                job.mark_failed(reason=f"Finalizing the job log failed: {exc}")
                state = "failed"
            if state not in {"done", "failed"}:
                continue

            del self._running[job]
            self.available_budget = self.available_budget.add(job.resources)
            self._release_allocations(job.assigned_cpu_indices, job.assigned_accelerator_indices)
            finished_any = True
            if state == "done":
                self._logger.info("LocalScheduler observed job completion for %s.", job.work_unit)
                runtime_job_done(id(job))
                self._release_dependents_locked(job)
            else:
                self._logger.error("LocalScheduler observed job failure for %s.", job.work_unit)
                runtime_job_failed(id(job))
                self._fail_dependents_locked(job)

        return finished_any

    def _start_ready_jobs_locked(self) -> None:
        while self._ready and self.available_budget.cpus and self.available_budget.memory:
            job = self._ready[0]
            if self._start_job_locked(job):
                self._ready.popleft()
            else:
                self._blocked.append(self._ready.popleft())

    def _retry_blocked_locked(self) -> None:
        while self._blocked and self._start_job_locked(self._blocked[0]):
            self._blocked.popleft()

    def _start_job_locked(self, job: LocalJob) -> bool:
        if not self.available_budget.fits(job.resources):
            return False
        allocations = self._reserve_indices(job.resources)
        if allocations is None:
            return False
        cpu_indices, accelerator_indices = allocations

        try:
            self._launch_job(job, cpu_indices=cpu_indices, accelerator_indices=accelerator_indices)
        except Exception as exc:
            self._logger.exception("Failed to launch local job for %s.", job.work_unit)
            self._release_allocations(cpu_indices, accelerator_indices)
            self._mark_failed_locked(job, reason=f"Launch failed: {type(exc).__name__}: {exc}")
            return True

        self.available_budget = self.available_budget.subtract(job.resources)
        self._running[job] = None
        return True

    def _release_dependents_locked(self, job: LocalJob) -> None:
        for dependent in self._dependents.pop(job, ()):
            remaining = self._waiting.get(dependent)
            if remaining is None:
                continue
            if remaining == 1:
                del self._waiting[dependent]
                self._ready.append(dependent)
            else:
                self._waiting[dependent] = remaining - 1

    def _fail_dependents_locked(self, job: LocalJob) -> None:
        failed = deque(self._dependents.pop(job, ()))
        while failed:
            dependent = failed.popleft()
            if self._waiting.pop(dependent, None) is None:
                continue
            dependent.mark_failed(reason="A dependency failed.")
            self._logger.error("Dependency failed; marked local job failed for %s.", dependent.work_unit)
            runtime_job_failed(id(dependent))
            failed.extend(self._dependents.pop(dependent, ()))

    def _mark_failed_locked(self, job: LocalJob, *, reason: str | None = None) -> None:
        job.mark_failed(reason=reason)
        self._logger.error("Marked pending local job failed for %s.", job.work_unit)
        runtime_job_failed(id(job))
        self._fail_dependents_locked(job)

    def _fail_all_locked(self, reason: str) -> None:
        """Move every tracked job to failed after a fatal scheduler error."""
        jobs = set(self._ready) | set(self._blocked) | set(self._waiting) | set(self._running)
        jobs.update(job for dependents in self._dependents.values() for job in dependents)
        for job in self._running:
            job.force_kill(reason=reason)
        for job in jobs:
            job.mark_failed(reason=reason)
            with contextlib.suppress(Exception):
                runtime_job_failed(id(job))
        for collection in (self._ready, self._blocked, self._waiting, self._running, self._dependents):
            collection.clear()

    def _launch_job(self, job: LocalJob, *, cpu_indices: list[int], accelerator_indices: list[int]) -> None:
        process: SupervisionSession | None = None
        try:
            prepare = job.snapshot.prepare_job if job.snapshot is not None else prepare_live_job
            job.job_id, argv, env_overrides, job.log_path = prepare(
                work_unit=job.work_unit,
                workspace=job.workspace,
            )
            self._logger.debug(
                "Launching local subprocess for %s with job_id=%s cpu_indices=%s accelerator_indices=%s log=%s.",
                job.work_unit,
                job.job_id,
                cpu_indices,
                accelerator_indices,
                job.log_path,
            )
            command = (
                Command(argv[0], argv[1:])
                .envs(
                    {
                        "FORCE_COLOR": "1",
                        "MISEN_RUNTIME_EVENTS": "1",
                    }
                    | env_overrides
                    | resource_environment(
                        cpu_indices,
                        self.available_budget.accelerator_type,
                        accelerator_indices,
                    )
                )
                .stdout_file(job.log_path, append=True)
                .stderr_file(job.log_path, append=True)
                .kill_on_parent_death()
            )
            if sys.platform in {"linux", "win32"}:
                command = command.cpu_affinity(cpu_indices)
            if self.enforce_time_limits:
                command = command.timeout(job.resources["time"] * 60).timeout_grace(_TIMEOUT_KILL_GRACE_S)
            process = Supervisor(command, restart="never").start()
            pid = job.set_process(
                process,
                cpu_indices=cpu_indices,
                accelerator_indices=accelerator_indices,
            )
        except BaseException as exc:
            if process is not None:
                try:
                    process.stop(0)
                except (ProcessError, RuntimeError) as cleanup_error:
                    exc.add_note(f"Additionally, stopping the partial local launch failed: {cleanup_error}")
            raise

        self._logger.info(
            "Launched local subprocess for %s (job_id=%s, pid=%s).",
            job.work_unit,
            job.job_id,
            pid or "pending",
        )
        runtime_job_running(id(job), job_id=job.job_id, pid=pid)

    def _reserve_indices(self, resources: Resources) -> tuple[list[int], list[int]] | None:
        cpu_count = resources["cpus"]
        if len(self.available_cpu_indices) < cpu_count:
            return None
        cpu_indices = self.available_cpu_indices[:cpu_count]
        del self.available_cpu_indices[:cpu_count]

        accelerator_count = resources["accelerators"] if resources["accelerator_type"] != "tpu" else 0
        if len(self.available_accelerator_indices) < accelerator_count:
            self._release_allocations(cpu_indices, [])
            return None
        accelerator_indices = self.available_accelerator_indices[:accelerator_count]
        del self.available_accelerator_indices[:accelerator_count]
        return cpu_indices, accelerator_indices

    def _release_allocations(self, cpu_indices: list[int], accelerator_indices: list[int]) -> None:
        for index in cpu_indices:
            insort(self.available_cpu_indices, index)
        for index in accelerator_indices:
            insort(self.available_accelerator_indices, index)

    def _terminate_running_jobs(self) -> None:
        """Gracefully stop and reap every running job during shutdown."""
        with self._condition:
            if self._shutting_down:
                return
            self._shutting_down = True
            jobs = list(self._running)
            pending = set(self._ready) | set(self._blocked) | set(self._waiting)
            pending.update(job for dependents in self._dependents.values() for job in dependents)
            pending.difference_update(self._running)
            reason = "Local executor shut down before the job completed."
            for job in pending:
                job.mark_failed(reason=reason)
                with contextlib.suppress(Exception):
                    runtime_job_failed(id(job))
            for collection in (self._ready, self._blocked, self._waiting, self._dependents):
                collection.clear()
            self._condition.notify_all()
        if jobs:
            self._logger.info("LocalScheduler terminating %d running job(s) at shutdown.", len(jobs))

        async def stop_jobs() -> list[bool | BaseException]:
            return await asyncio.gather(
                *(
                    job.astop(
                        grace_seconds=_SHUTDOWN_TERM_GRACE_S,
                        reason=reason,
                    )
                    for job in jobs
                ),
                return_exceptions=True,
            )

        for job, result in zip(jobs, asyncio.run(stop_jobs()), strict=True):
            if isinstance(result, BaseException):
                self._logger.error("Error terminating job %s during shutdown: %s", job.work_unit, result)
            with contextlib.suppress(Exception):
                runtime_job_failed(id(job))
        with self._condition:
            self._running.clear()


def _install_sigterm_handler(cleanup: Callable[[], None]) -> None:
    """Schedule lock-taking cleanup off the signal handler, then propagate SIGTERM."""
    sigterm = getattr(signal, "SIGTERM", None)
    if sigterm is None:
        return
    try:
        previous = signal.getsignal(sigterm)
    except ValueError:
        return

    cleanup_requested, cleanup_done = threading.Event(), threading.Event()
    propagate_default = not callable(previous) and previous != signal.SIG_IGN

    def cleanup_worker() -> None:
        cleanup_requested.wait()
        try:
            cleanup()
        except BaseException:
            logger.exception("Error terminating local jobs after SIGTERM.")
        finally:
            cleanup_done.set()

    def propagate() -> None:
        cleanup_requested.wait()
        cleanup_done.wait(_SIGTERM_CLEANUP_TIMEOUT_S)
        if propagate_default:
            os.kill(os.getpid(), sigterm)

    def handler(signum: int, frame: FrameType | None) -> None:
        if propagate_default:
            signal.signal(signum, signal.SIG_DFL)
        cleanup_requested.set()
        if callable(previous):
            cast("Callable[[int, FrameType | None], Any]", previous)(signum, frame)

    try:
        signal.signal(sigterm, handler)
    except ValueError:
        return
    threading.Thread(target=cleanup_worker, daemon=True, name="misen-sigterm-cleanup").start()
    threading.Thread(target=propagate, daemon=True, name="misen-sigterm-propagation").start()
