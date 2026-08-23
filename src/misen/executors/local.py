"""Local subprocess-based executor implementation."""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from bisect import insort
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from misen.exceptions import ExecutionError, StorageError, SubmissionError
from misen.executor import Executor, Job
from misen.task_metadata import AcceleratorType  # noqa: TC001  # needed by msgspec during config decoding
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
    from io import FileIO
    from types import FrameType

    from misen.task_metadata import Resources
    from misen.utils.snapshot import ProjectSnapshot
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = ("LocalExecutor", "LocalJob")

_JobState = Literal["pending", "running", "done", "failed"]
_ProcessSignal = Literal["terminate", "kill"]
logger = logging.getLogger(__name__)
_TIMEOUT_KILL_GRACE_S = 5.0
_FATAL_REAP_TIMEOUT_S = 2.0
_SHUTDOWN_TERM_GRACE_S = 2.0
_SIGTERM_CLEANUP_TIMEOUT_S = 5.0


def _signal_process_group(process: subprocess.Popen[bytes], action: _ProcessSignal) -> OSError | None:
    """Signal a local worker and its descendants, with a portable fallback."""
    if os.name == "posix" and isinstance(process, subprocess.Popen):
        sig = signal.SIGTERM if action == "terminate" else signal.SIGKILL
        try:
            os.killpg(process.pid, sig)
        except OSError:
            pass
        else:
            return None
    try:
        getattr(process, action)()
    except OSError as exc:
        return exc
    return None


def _start_process_reaper(process: subprocess.Popen[bytes]) -> None:
    """Keep killing and waiting for a child that fatal recovery could not reap."""

    def reap() -> None:
        while process.poll() is None:
            _signal_process_group(process, "kill")
            try:
                process.wait(timeout=1.0)
            except (OSError, subprocess.TimeoutExpired):
                time.sleep(0.1)

    threading.Thread(target=reap, daemon=True, name=f"misen-process-reaper[{process.pid}]").start()


class LocalJob(Job):
    """Job handle backed by a local subprocess."""

    __slots__ = (
        "_cached_state",
        "_lock",
        "_log_fp",
        "_process",
        "_started_at",
        "_timed_out_at",
        "_timeout_kill_sent",
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
        self._process: subprocess.Popen[bytes] | None = None
        self._log_fp: FileIO | None = None
        self._cached_state: _JobState = "pending"
        self._started_at: float | None = None
        self._timed_out_at: float | None = None
        self._timeout_kill_sent = False
        self._lock = threading.Lock()

    def state(self) -> _JobState:
        """Return the current process-backed job state."""
        with self._lock:
            if self._cached_state in {"done", "failed"}:
                self._finalize_job_log_locked()
                return self._cached_state
            if self._process is None:
                return "pending"

            return_code = self._process.poll()
            if return_code is None:
                return "running"

            # No descendant may outlive the worker process-group leader.
            _signal_process_group(self._process, "kill")
            close_error = self._close_log_fp_locked()
            self._cached_state = "done" if return_code == 0 and self._timed_out_at is None else "failed"
            if return_code != 0 and self._timed_out_at is None:
                self._record_failure(f"Process exited with code {return_code}.")
            if close_error is not None:
                self._cached_state = "failed"
                self._record_failure(f"Closing the local job log failed: {close_error}")
            self._finalize_job_log_locked()
            logger.info(
                "Local job %s for %s exited with code %d -> state=%s.",
                self.job_id or "n/a",
                self.label,
                return_code,
                self._cached_state,
            )
            return self._cached_state

    def _finalize_job_log_locked(self) -> None:
        """Publish the terminal log once the worker process has exited."""
        if self._process is not None and self._process.poll() is None:
            return
        self._finalize_log(self.workspace, failed=self._cached_state == "failed")

    def set_process(
        self,
        process: subprocess.Popen[bytes],
        *,
        log_fp: FileIO,
        cpu_indices: list[int],
        accelerator_indices: list[int],
    ) -> None:
        """Attach subprocess/log handles and mark the job running."""
        with self._lock:
            self._process = process
            self._log_fp = log_fp
            self.assigned_cpu_indices = list(cpu_indices)
            self.assigned_accelerator_indices = list(accelerator_indices)
            self._cached_state = "running"
            self._started_at = time.monotonic()
            logger.info(
                "Local job %s for %s is running (pid=%d).",
                self.job_id or "n/a",
                self.label,
                process.pid,
            )

    def time_limit_exceeded(self) -> bool:
        """Return True if a running job has exceeded its requested time limit."""
        with self._lock:
            if self._started_at is None or self._cached_state != "running":
                return False
            return time.monotonic() - self._started_at > self.resources["time"] * 60

    def mark_failed(self, reason: str | None = None) -> None:
        """Mark a pending/running job failed and close local handles."""
        with self._lock:
            close_error = self._close_log_fp_locked()
            if reason is not None:
                self._record_failure(reason)
            if close_error is not None:
                self._record_failure(f"Additionally, closing the local job log failed: {close_error}")
            self._cached_state = "failed"
            logger.error("Local job %s for %s marked failed.", self.job_id or "n/a", self.label)

    def terminate(self, *, reason: str | None = None) -> None:
        """Send SIGTERM to the subprocess if it is still running."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return
            if reason is not None:
                self._record_failure(reason)
            if _signal_process_group(self._process, "terminate") is not None:
                return
            logger.info(
                "Local job %s for %s sent SIGTERM (pid=%d).",
                self.job_id or "n/a",
                self.label,
                self._process.pid,
            )

    def force_kill(self, *, reason: str | None = None, wait_timeout_s: float = _FATAL_REAP_TIMEOUT_S) -> bool:
        """Kill and reap a subprocess, handing stubborn processes to a watchdog."""
        with self._lock:
            if self._process is None:
                return True
            process = self._process
            leader_running = process.poll() is None
            if reason is not None:
                self._record_failure(reason)
            if _signal_process_group(process, "kill") is not None:
                if leader_running:
                    _start_process_reaper(process)
                    return False
                return True
        if not leader_running:
            return True
        try:
            process.wait(timeout=wait_timeout_s)
        except (OSError, subprocess.TimeoutExpired):
            _start_process_reaper(process)
            return False
        return True

    def enforce_timeout(self, *, kill_grace_s: float = _TIMEOUT_KILL_GRACE_S) -> Literal["terminate", "kill"] | None:
        """Enforce the requested limit once, escalating to SIGKILL after a grace period."""
        with self._lock:
            if self._started_at is None or self._cached_state != "running" or self._process is None:
                return None
            if self._process.poll() is not None:
                return None

            now = time.monotonic()
            if self._timed_out_at is None:
                if now - self._started_at <= self.resources["time"] * 60:
                    return None
                self._timed_out_at = now
                self._record_failure(f"Exceeded the requested {self.resources['time']} minute time limit.")
                if _signal_process_group(self._process, "terminate") is not None:
                    return None
                return "terminate"

            if self._timeout_kill_sent or now - self._timed_out_at < kill_grace_s:
                return None
            if _signal_process_group(self._process, "kill") is not None:
                return None
            self._timeout_kill_sent = True
            return "kill"

    def _close_log_fp_locked(self) -> OSError | None:
        if self._log_fp is None:
            return None
        log_fp = self._log_fp
        self._log_fp = None
        try:
            log_fp.close()
        except OSError as exc:
            logger.exception("Could not close the local job log for %s.", self.label)
            return exc
        return None


class LocalExecutor(Executor[LocalJob]):
    """Executor that runs work units as local subprocesses.

    Snapshots are content-addressed project state published to the
    workspace; environments materialize from them into the env store at
    ``env_store_dir`` (default ``/tmp/misen-env-store-<user>``). With
    ``prewarm_envs`` (the default here) they are built once at submission —
    concurrent jobs share them and env failures surface before any job
    starts; with ``prewarm_envs=False`` the first job builds them at
    startup instead.
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
        """Infer resource limits and initialize the scheduler."""
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
        if self.cpu_indices is None:
            if self.num_cpus == "all":
                cpu_indices = list(range(os.cpu_count() or 1))
            elif isinstance(self.num_cpus, bool) or not isinstance(self.num_cpus, int) or self.num_cpus < 1:
                msg = "num_cpus must be 'all' or a positive integer."
                raise ValueError(msg)
            else:
                cpu_indices = list(range(self.num_cpus))
        else:
            if not self.cpu_indices or any(
                isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in self.cpu_indices
            ):
                msg = "cpu_indices must contain nonnegative integer CPU indices."
                raise ValueError(msg)
            cpu_indices = sorted(set(self.cpu_indices))

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
                "Requested resources exceed LocalExecutor limits: "
                f"requested cpus={resources['cpus']}, memory={resources['memory']}, "
                f"accelerators={resources['accelerators']} (type={resources['accelerator_type']}); "
                f"limits cpus={self._resource_budget.cpus}, memory={self._resource_budget.memory}, "
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
        self._condition = threading.Condition()
        self._thread = threading.Thread(name="misen-local-scheduler", target=self._run, daemon=True)
        self._thread.start()
        atexit.register(self._terminate_running_jobs)
        _install_sigterm_handler(self._terminate_running_jobs)
        self._logger.info("Started LocalScheduler background thread.")

    def submit(self, job: LocalJob) -> None:
        """Queue a job for scheduling."""
        with self._condition:
            if self._fatal_error is not None:
                msg = f"Local scheduler is unavailable after {type(self._fatal_error).__name__}: {self._fatal_error}"
                raise ExecutionError(msg) from self._fatal_error
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

    def _run(self) -> None:
        while True:
            try:
                with self._condition:
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
        if self.enforce_time_limits:
            self._terminate_timed_out_locked()
        if self._collect_finished_locked():
            self._retry_blocked_locked()
        self._start_ready_jobs_locked()

    def _terminate_timed_out_locked(self) -> None:
        for job in list(self._running):
            action = job.enforce_timeout()
            if action is not None:
                self._logger.warning(
                    "Local job %s for %s exceeded its time limit; sent %s.",
                    job.job_id or "n/a",
                    job.label,
                    "SIGTERM" if action == "terminate" else "SIGKILL",
                )

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
        log_fp: FileIO | None = None
        process: subprocess.Popen[bytes] | None = None
        try:
            prepare = job.snapshot.prepare_job if job.snapshot is not None else prepare_live_job
            job.job_id, argv, env_overrides, job.log_path = prepare(
                work_unit=job.work_unit,
                workspace=job.workspace,
                cpu_indices=cpu_indices,
                accelerator_type=self.available_budget.accelerator_type,
                accelerator_indices=accelerator_indices,
            )
            log_fp = job.log_path.open("ab", buffering=0)
            self._logger.debug(
                "Launching local subprocess for %s with job_id=%s cpu_indices=%s accelerator_indices=%s log=%s.",
                job.work_unit,
                job.job_id,
                cpu_indices,
                accelerator_indices,
                job.log_path,
            )
            process = subprocess.Popen(  # noqa: S603
                argv,
                env=os.environ
                | {
                    "FORCE_COLOR": "1",
                    "MISEN_RUNTIME_EVENTS": "1",
                }
                | env_overrides,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                preexec_fn=_PREEXEC_FN,  # noqa: PLW1509
                start_new_session=os.name == "posix",
            )
            job.set_process(
                process,
                log_fp=log_fp,
                cpu_indices=cpu_indices,
                accelerator_indices=accelerator_indices,
            )
        except BaseException as exc:
            if process is not None and process.poll() is None:  # noqa: SIM102 - preserve cleanup error
                if cleanup_error := _signal_process_group(process, "terminate"):
                    exc.add_note(f"Additionally, terminating the partial local launch failed: {cleanup_error}")
            if log_fp is not None:
                try:
                    log_fp.close()
                except OSError as cleanup_error:
                    exc.add_note(f"Additionally, closing the partial local job log failed: {cleanup_error}")
            raise

        self._logger.info(
            "Launched local subprocess for %s (job_id=%s, pid=%d).",
            job.work_unit,
            job.job_id,
            process.pid,
        )
        runtime_job_running(id(job), job_id=job.job_id, pid=process.pid)

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
        """Terminate, then kill and reap every running job during shutdown."""
        with self._condition:
            jobs = list(self._running)
        if jobs:
            self._logger.info("LocalScheduler terminating %d running job(s) at shutdown.", len(jobs))
        for job in jobs:
            try:
                job.terminate()
            except Exception:
                self._logger.exception("Error terminating job %s during shutdown.", job.work_unit)

        def still_running(job: LocalJob) -> bool:
            try:
                return job.state() == "running"
            except Exception:
                self._logger.exception("Error polling job %s during shutdown.", job.work_unit)
                return True

        deadline = time.monotonic() + _SHUTDOWN_TERM_GRACE_S
        while time.monotonic() < deadline and any(still_running(job) for job in jobs):
            time.sleep(0.05)
        for job in jobs:
            job.force_kill(reason="Local executor shut down before the job completed.")


def _build_preexec_fn() -> Callable[[], None] | None:
    """Return a Linux-only preexec hook that SIGTERMs children on parent death."""
    if sys.platform != "linux":
        return None
    try:
        import ctypes
        import ctypes.util

        prctl = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True).prctl
    except (OSError, AttributeError):
        return None

    pr_set_pdeathsig = 1

    def _set_pdeathsig() -> None:
        prctl(pr_set_pdeathsig, signal.SIGTERM, 0, 0, 0)
        if os.getppid() == 1:
            os.kill(os.getpid(), signal.SIGTERM)

    return _set_pdeathsig


_PREEXEC_FN = _build_preexec_fn()


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
