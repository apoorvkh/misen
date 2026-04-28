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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from misen.executor import Executor, Job
from misen.utils.assigned_resources import AssignedResources
from misen.utils.runtime_events import (
    runtime_job_done,
    runtime_job_failed,
    runtime_job_pending,
    runtime_job_running,
    task_label,
    work_unit_label,
)
from misen.utils.snapshot import LocalSnapshot, NullSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable
    from io import FileIO
    from types import FrameType

    from misen.task_metadata import GpuRuntime, Resources
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = ("LocalExecutor", "LocalJob")

_JobState = Literal["pending", "running", "done", "failed"]
logger = logging.getLogger(__name__)


class LocalJob(Job):
    """Job handle backed by a local subprocess."""

    __slots__ = (
        "_cached_state",
        "_lock",
        "_log_fp",
        "_process",
        "_started_at",
        "assigned_cpu_indices",
        "assigned_gpu_indices",
        "dependencies",
        "snapshot",
        "workspace",
    )

    def __init__(
        self,
        work_unit: WorkUnit,
        dependencies: set[LocalJob],
        snapshot: LocalSnapshot | NullSnapshot,
        workspace: Workspace,
    ) -> None:
        """Initialize a local job."""
        super().__init__(work_unit=work_unit, job_id=None, log_path=None)
        self.dependencies = set(dependencies)
        self.snapshot = snapshot
        self.workspace = workspace
        self.assigned_cpu_indices: list[int] = []
        self.assigned_gpu_indices: list[int] = []
        self._process: subprocess.Popen[bytes] | None = None
        self._log_fp: FileIO | None = None
        self._cached_state: _JobState = "pending"
        self._started_at: float | None = None
        self._lock = threading.Lock()

    def state(self) -> _JobState:
        """Return the current process-backed job state."""
        with self._lock:
            if self._cached_state in {"done", "failed"}:
                return self._cached_state
            if self._process is None:
                return "pending"

            return_code = self._process.poll()
            if return_code is None:
                return "running"

            self._close_log_fp_locked()
            self._cached_state = "done" if return_code == 0 else "failed"
            logger.info(
                "Local job %s for %s exited with code %d -> state=%s.",
                self.job_id or "n/a",
                self.label,
                return_code,
                self._cached_state,
            )
            return self._cached_state

    def set_process(
        self,
        process: subprocess.Popen[bytes],
        *,
        log_fp: FileIO,
        cpu_indices: list[int],
        gpu_indices: list[int],
    ) -> None:
        """Attach subprocess/log handles and mark the job running."""
        with self._lock:
            self._process = process
            self._log_fp = log_fp
            self.assigned_cpu_indices = list(cpu_indices)
            self.assigned_gpu_indices = list(gpu_indices)
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
        time_limit = self.resources["time"]
        if time_limit is None:
            return False
        with self._lock:
            if self._started_at is None or self._cached_state != "running":
                return False
            return time.monotonic() - self._started_at > time_limit * 60

    def mark_failed(self) -> None:
        """Mark a pending/running job failed and close local handles."""
        with self._lock:
            self._close_log_fp_locked()
            self._cached_state = "failed"
            logger.error("Local job %s for %s marked failed.", self.job_id or "n/a", self.label)

    def terminate(self) -> None:
        """Send SIGTERM to the subprocess if it is still running."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return
            try:
                self._process.terminate()
            except OSError:
                return
            logger.info(
                "Local job %s for %s sent SIGTERM (pid=%d).",
                self.job_id or "n/a",
                self.label,
                self._process.pid,
            )

    def _close_log_fp_locked(self) -> None:
        if self._log_fp is None:
            return
        try:
            self._log_fp.close()
        finally:
            self._log_fp = None


class LocalExecutor(Executor[LocalJob, "LocalSnapshot | NullSnapshot"]):
    """Executor that runs work units as local subprocesses."""

    max_memory: int | Literal["all"] = "all"
    num_cpus: int | Literal["all"] = "all"
    cpu_indices: list[int] | None = None
    num_cuda_gpus: int = 0
    cuda_gpu_indices: list[int] | None = None
    num_rocm_gpus: int = 0
    rocm_gpu_indices: list[int] | None = None
    num_xpu_gpus: int = 0
    xpu_gpu_indices: list[int] | None = None
    snapshot: bool = True
    enforce_time_limits: bool = False

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

        gpu_indices_by_runtime: dict[str, list[int]] = {}
        for runtime, count, indices in (
            ("cuda", self.num_cuda_gpus, self.cuda_gpu_indices),
            ("rocm", self.num_rocm_gpus, self.rocm_gpu_indices),
            ("xpu", self.num_xpu_gpus, self.xpu_gpu_indices),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                msg = f"num_{runtime}_gpus must be a nonnegative integer."
                raise ValueError(msg)
            if indices is not None:
                if count:
                    msg = f"num_{runtime}_gpus and {runtime}_gpu_indices should not both be passed to LocalExecutor."
                    raise ValueError(msg)
                if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices):
                    msg = f"{runtime}_gpu_indices must contain nonnegative integer GPU indices."
                    raise ValueError(msg)
                gpu_indices_by_runtime[runtime] = sorted(set(indices))
            else:
                gpu_indices_by_runtime[runtime] = list(range(count))

        self._resource_budget = _ResourceBudget(
            memory=self.max_memory,
            cpus=len(cpu_indices),
            cuda_gpus=len(gpu_indices_by_runtime["cuda"]),
            rocm_gpus=len(gpu_indices_by_runtime["rocm"]),
            xpu_gpus=len(gpu_indices_by_runtime["xpu"]),
        )
        self._scheduler = _LocalScheduler(
            available_budget=self._resource_budget,
            available_cpu_indices=cpu_indices,
            available_gpu_indices=gpu_indices_by_runtime,
            enforce_time_limits=self.enforce_time_limits,
        )
        logger.info(
            "Initialized LocalExecutor budget: memory=%sGiB cpus=%d cuda_gpus=%d rocm_gpus=%d xpu_gpus=%d.",
            self._resource_budget.memory,
            self._resource_budget.cpus,
            self._resource_budget.cuda_gpus,
            self._resource_budget.rocm_gpus,
            self._resource_budget.xpu_gpus,
        )

    def _make_snapshot(self, workspace: Workspace) -> LocalSnapshot | NullSnapshot:
        """Return a local snapshot for this workspace, or ``NullSnapshot`` when disabled."""
        return self._make_local_snapshot(workspace=workspace) if self.snapshot else NullSnapshot()

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[LocalJob],
        workspace: Workspace,
        snapshot: LocalSnapshot | NullSnapshot,
    ) -> LocalJob:
        """Queue a work unit in the local scheduler."""
        resources = work_unit.resources
        if not self._resource_budget.fits(resources):
            msg = (
                "Requested resources exceed LocalExecutor limits: "
                f"requested cpus={resources['cpus']}, memory={resources['memory']}, "
                f"gpus={resources['gpus']} (runtime={resources['gpu_runtime']}); "
                f"limits cpus={self._resource_budget.cpus}, memory={self._resource_budget.memory}, "
                f"cuda_gpus={self._resource_budget.cuda_gpus}, "
                f"rocm_gpus={self._resource_budget.rocm_gpus}, "
                f"xpu_gpus={self._resource_budget.xpu_gpus}."
            )
            raise ValueError(msg)

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
    cuda_gpus: int
    rocm_gpus: int
    xpu_gpus: int

    def fits(self, resources: Resources) -> bool:
        match resources["gpu_runtime"]:
            case "cuda":
                gpus = self.cuda_gpus
            case "rocm":
                gpus = self.rocm_gpus
            case "xpu":
                gpus = self.xpu_gpus
            case runtime:
                msg = f"Unsupported GPU runtime: {runtime!r}."
                raise ValueError(msg)
        return resources["cpus"] <= self.cpus and resources["memory"] <= self.memory and resources["gpus"] <= gpus

    def add(self, resources: Resources) -> _ResourceBudget:
        return self._adjust(resources, 1)

    def subtract(self, resources: Resources) -> _ResourceBudget:
        return self._adjust(resources, -1)

    def _adjust(self, resources: Resources, sign: Literal[-1, 1]) -> _ResourceBudget:
        runtime = resources["gpu_runtime"]
        gpu_delta = resources["gpus"] * sign
        return _ResourceBudget(
            memory=self.memory + resources["memory"] * sign,
            cpus=self.cpus + resources["cpus"] * sign,
            cuda_gpus=self.cuda_gpus + (gpu_delta if runtime == "cuda" else 0),
            rocm_gpus=self.rocm_gpus + (gpu_delta if runtime == "rocm" else 0),
            xpu_gpus=self.xpu_gpus + (gpu_delta if runtime == "xpu" else 0),
        )


class _LocalScheduler:
    """Background scheduler for dependency-ready local jobs."""

    __slots__ = (
        "_condition",
        "_pending",
        "_running",
        "_thread",
        "available_budget",
        "available_cpu_indices",
        "available_gpu_indices",
        "enforce_time_limits",
    )
    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        available_budget: _ResourceBudget,
        available_cpu_indices: list[int],
        available_gpu_indices: dict[str, list[int]],
        enforce_time_limits: bool = False,
    ) -> None:
        self.available_budget = available_budget
        self.available_cpu_indices = list(available_cpu_indices)
        self.available_gpu_indices = {runtime: list(indices) for runtime, indices in available_gpu_indices.items()}
        self.enforce_time_limits = enforce_time_limits
        self._pending: list[LocalJob] = []
        self._running: set[LocalJob] = set()
        self._condition = threading.Condition()
        self._thread = threading.Thread(name="misen-local-scheduler", target=self._run, daemon=True)
        self._thread.start()
        atexit.register(self._terminate_running_jobs)
        _install_sigterm_handler(self._terminate_running_jobs)
        self._logger.info("Started LocalScheduler background thread.")

    def submit(self, job: LocalJob) -> None:
        """Queue a job for scheduling."""
        with self._condition:
            self._pending.append(job)
            self._logger.debug(
                "Queued job for %s (pending=%d, running=%d).",
                job.work_unit,
                len(self._pending),
                len(self._running),
            )
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                if self.enforce_time_limits:
                    self._terminate_timed_out_locked()
                self._collect_finished_locked()
                progress_made = self._start_ready_jobs_locked()
                if not self._pending and not self._running:
                    self._condition.wait()
                elif not progress_made:
                    self._condition.wait(timeout=0.1)

    def _terminate_timed_out_locked(self) -> None:
        for job in list(self._running):
            if not job.time_limit_exceeded():
                continue
            self._logger.warning(
                "Local job %s for %s exceeded time limit of %d minute(s); sending SIGTERM.",
                job.job_id or "n/a",
                job.label,
                job.resources["time"],
            )
            job.terminate()

    def _collect_finished_locked(self) -> None:
        finished_any = False
        for job in list(self._running):
            state = job.state()
            if state not in {"done", "failed"}:
                continue

            self._running.remove(job)
            self.available_budget = self.available_budget.add(job.resources)
            self._release_allocations(
                job.assigned_cpu_indices,
                job.resources["gpu_runtime"],
                job.assigned_gpu_indices,
            )
            finished_any = True
            if state == "done":
                self._logger.info("LocalScheduler observed job completion for %s.", job.work_unit)
                runtime_job_done(id(job))
            else:
                self._logger.error("LocalScheduler observed job failure for %s.", job.work_unit)
                runtime_job_failed(id(job))

        if finished_any:
            self._condition.notify_all()

    def _start_ready_jobs_locked(self) -> bool:
        progress_made = False
        for job in list(self._pending):
            dependency_states = {dependency.state() for dependency in job.dependencies}
            if "failed" in dependency_states:
                self._logger.error(
                    "Dependency failed; marking pending job failed for %s.",
                    job.work_unit,
                )
                self._mark_pending_failed(job)
                progress_made = True
                continue
            if dependency_states and dependency_states != {"done"}:
                continue
            if not self.available_budget.fits(job.resources):
                continue

            allocations = self._reserve_indices(job.resources)
            if allocations is None:
                continue
            cpu_indices, gpu_indices = allocations

            try:
                self._launch_job(job, cpu_indices=cpu_indices, gpu_indices=gpu_indices)
            except Exception:
                self._logger.exception("Failed to launch local job for %s.", job.work_unit)
                self._mark_pending_failed(job, cpu_indices=cpu_indices, gpu_indices=gpu_indices)
                progress_made = True
                continue

            self.available_budget = self.available_budget.subtract(job.resources)
            self._pending.remove(job)
            self._running.add(job)
            progress_made = True
        return progress_made

    def _launch_job(self, job: LocalJob, *, cpu_indices: list[int], gpu_indices: list[int]) -> None:
        assigned_resources = AssignedResources(
            cpu_indices=cpu_indices,
            gpu_indices=gpu_indices,
            memory=job.resources["memory"],
            gpu_memory=job.resources["gpu_memory"],
        )
        log_fp: FileIO | None = None
        process: subprocess.Popen[bytes] | None = None
        try:
            job.job_id, argv, env_overrides, job.log_path = job.snapshot.prepare_job(
                work_unit=job.work_unit,
                workspace=job.workspace,
                assigned_resources_getter=lambda r=assigned_resources: r,
                gpu_runtime=job.resources["gpu_runtime"],
            )
            log_fp = job.log_path.open("ab", buffering=0)
            self._logger.debug(
                "Launching local subprocess for %s with job_id=%s cpu_indices=%s gpu_indices=%s log=%s.",
                job.work_unit,
                job.job_id,
                cpu_indices,
                gpu_indices,
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
            )
            job.set_process(process, log_fp=log_fp, cpu_indices=cpu_indices, gpu_indices=gpu_indices)
        except Exception:
            if process is not None and process.poll() is None:
                with contextlib.suppress(OSError):
                    process.terminate()
            if log_fp is not None:
                log_fp.close()
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

        gpu_pool = self.available_gpu_indices[resources["gpu_runtime"]]
        gpu_count = resources["gpus"]
        if len(gpu_pool) < gpu_count:
            for index in cpu_indices:
                insort(self.available_cpu_indices, index)
            return None
        gpu_indices = gpu_pool[:gpu_count]
        del gpu_pool[:gpu_count]
        return cpu_indices, gpu_indices

    def _release_allocations(self, cpu_indices: list[int], gpu_runtime: GpuRuntime, gpu_indices: list[int]) -> None:
        for index in cpu_indices:
            insort(self.available_cpu_indices, index)
        for index in gpu_indices:
            insort(self.available_gpu_indices[gpu_runtime], index)

    def _mark_pending_failed(
        self,
        job: LocalJob,
        *,
        cpu_indices: list[int] | None = None,
        gpu_indices: list[int] | None = None,
    ) -> None:
        self._release_allocations(cpu_indices or [], job.resources["gpu_runtime"], gpu_indices or [])
        job.mark_failed()
        if job in self._pending:
            self._pending.remove(job)
        self._logger.error("Marked pending local job failed for %s.", job.work_unit)
        runtime_job_failed(id(job))

    def _terminate_running_jobs(self) -> None:
        """Send SIGTERM to every currently running job.

        Registered via :mod:`atexit` so local subprocesses do not outlive the
        parent Python process on normal shutdown / KeyboardInterrupt. For hard
        parent death (SIGKILL, segfault), Linux children are also covered by
        ``prctl(PR_SET_PDEATHSIG)`` installed in ``preexec_fn`` at launch.
        """
        with self._condition:
            jobs = list(self._running)
        if jobs:
            self._logger.info("LocalScheduler terminating %d running job(s) at shutdown.", len(jobs))
        for job in jobs:
            try:
                job.terminate()
            except Exception:
                self._logger.exception("Error terminating job %s during shutdown.", job.work_unit)


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
    """Run ``cleanup`` on SIGTERM, then defer to the previous handler."""
    sigterm = getattr(signal, "SIGTERM", None)
    if sigterm is None:
        return
    try:
        previous = signal.getsignal(sigterm)
    except ValueError:
        return

    def handler(signum: int, frame: FrameType | None) -> None:
        try:
            cleanup()
        finally:
            if callable(previous):
                cast("Callable[[int, FrameType | None], Any]", previous)(signum, frame)
            elif previous != signal.SIG_IGN:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

    try:
        signal.signal(sigterm, handler)
    except ValueError:
        return
