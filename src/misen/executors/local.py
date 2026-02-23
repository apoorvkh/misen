"""Local subprocess-based executor implementation.

Concurrency model:

- Submit-time builds a dependency graph of work units.
- A background scheduler thread launches ready jobs when resources fit.
- Resource budgeting is explicit (CPUs, memory, runtime-specific GPUs) and
  prevents oversubscription.
- Dependency failures propagate to downstream pending jobs.

Execution still uses the generic executor contract (snapshot + dispatch + job
handles), but this module performs all scheduling in-process.
"""

from __future__ import annotations

import os
import subprocess
import threading
from bisect import insort
from functools import cache
from typing import TYPE_CHECKING, Literal

import msgspec

from misen.executor import Executor, Job
from misen.utils.assigned_resources import AssignedResources, build_assigned_resources_env
from misen.utils.runtime_events import (
    runtime_job_done,
    runtime_job_failed,
    runtime_job_pending,
    runtime_job_running,
    work_unit_label,
)
from misen.utils.snapshot import LocalSnapshot

if TYPE_CHECKING:
    from io import FileIO
    from pathlib import Path

    from misen.task_properties import GpuRuntime, Resources
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

JobState = Literal["pending", "running", "done", "failed", "unknown"]


class _ResourceBudget(msgspec.Struct, frozen=True):
    """Available resource budget for local scheduling.

    Values are interpreted as hard scheduler limits for concurrent job
    placement.
    """

    memory: int
    cpus: int
    cuda_gpus: int
    rocm_gpus: int
    xpu_gpus: int

    def fits(self, resources: Resources) -> bool:
        """Return whether requested resources fit in current budget."""
        return (
            resources.cpus <= self.cpus
            and resources.memory <= self.memory
            and resources.gpus <= self._runtime_gpu_budget(resources.gpu_runtime)
        )

    def subtract(self, resources: Resources) -> _ResourceBudget:
        """Return new budget after reserving resources."""
        return self._adjust(resources=resources, multiplier=-1)

    def add(self, resources: Resources) -> _ResourceBudget:
        """Return new budget after releasing resources."""
        return self._adjust(resources=resources, multiplier=1)

    def _adjust(self, resources: Resources, multiplier: Literal[-1, 1]) -> _ResourceBudget:
        runtime_delta = resources.gpus * multiplier
        return _ResourceBudget(
            cpus=self.cpus + (resources.cpus * multiplier),
            memory=self.memory + (resources.memory * multiplier),
            cuda_gpus=self.cuda_gpus + (runtime_delta if resources.gpu_runtime == "cuda" else 0),
            rocm_gpus=self.rocm_gpus + (runtime_delta if resources.gpu_runtime == "rocm" else 0),
            xpu_gpus=self.xpu_gpus + (runtime_delta if resources.gpu_runtime == "xpu" else 0),
        )

    def _runtime_gpu_budget(self, gpu_runtime: GpuRuntime) -> int:
        match gpu_runtime:
            case "cuda":
                return self.cuda_gpus
            case "rocm":
                return self.rocm_gpus
            case "xpu":
                return self.xpu_gpus


class LocalJob(Job):
    """Job handle backed by a ``subprocess.Popen`` child process."""

    __slots__ = (
        "_lock",
        "_log_fp",
        "_process",
        "_state",
        "assigned_cpu_indices",
        "assigned_gpu_indices",
        "dependencies",
        "resources",
        "snapshot",
        "workspace",
    )

    def __init__(
        self,
        work_unit: WorkUnit,
        resources: Resources,
        dependencies: set[LocalJob],
        snapshot: LocalSnapshot,
        workspace: Workspace,
    ) -> None:
        """Initialize a local job."""
        super().__init__(work_unit=work_unit, job_id=None, log_path=None)
        self.resources = resources
        self.dependencies = dependencies
        self.snapshot = snapshot
        self.workspace = workspace

        self.assigned_cpu_indices: list[int] = []
        self.assigned_gpu_indices: list[int] = []

        self._process: subprocess.Popen[bytes] | None = None
        self._log_fp = None  # keep alive while the child runs
        self._state: JobState = "pending"
        self._lock = threading.Lock()

    def state(self) -> JobState:
        """Return current process-backed job state."""
        with self._lock:
            if self._state in {"done", "failed"}:
                return self._state

            proc = self._process
            if proc is None:
                return "pending"

            rc = proc.poll()
            if rc is None:
                return "running"

            # Process exited; close log handle.
            self._close_log_fp_locked()

            self._state = "done" if rc == 0 else "failed"
            return self._state

    def set_process(self, process: subprocess.Popen[bytes], log_fp: FileIO) -> None:
        """Attach process/log handles and mark job as running."""
        with self._lock:
            self._process = process
            self._log_fp = log_fp
            self._state = "running"

    def mark_failed(self) -> None:
        """Mark pending/running job as failed and close log handles."""
        with self._lock:
            self._close_log_fp_locked()
            self._state = "failed"

    def _close_log_fp_locked(self) -> None:
        fp = self._log_fp
        if fp is None:
            return
        try:
            fp.close()
        finally:
            self._log_fp = None


class _LocalScheduler(msgspec.Struct, dict=True):
    """Background scheduler for local jobs.

    The scheduler owns:

    - pending/running queues
    - resource allocation state
    - dependency readiness checks

    It is intentionally isolated from :class:`LocalExecutor` so executor
    dispatch stays lightweight and thread-safe.
    """

    available_budget: _ResourceBudget
    available_cpu_indices: list[int]
    available_cuda_gpu_indices: list[int]
    available_rocm_gpu_indices: list[int]
    available_xpu_gpu_indices: list[int]

    def __post_init__(self) -> None:
        self._pending: list[LocalJob] = []
        self._running: set[LocalJob] = set()
        self._condition = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, job: LocalJob) -> None:
        """Queue a job for scheduling."""
        with self._condition:
            self._pending.append(job)
            self._condition.notify_all()

    def _run(self) -> None:
        """Scheduler loop that launches ready jobs and reaps finished ones."""
        while True:
            with self._condition:
                self._collect_finished_locked()
                started_any = self._start_ready_jobs_locked()

                if not self._pending and not self._running:
                    self._condition.wait()
                    continue

                # If we didn't start anything, sleep briefly (we'll be notified on submit).
                if not started_any:
                    self._condition.wait(timeout=0.1)

    def _collect_finished_locked(self) -> None:
        """Reclaim resources from terminal jobs.

        Notes:
            Caller must hold ``self._condition``.
        """
        finished: list[tuple[LocalJob, JobState]] = []
        for job in self._running:
            state = job.state()
            if state in {"done", "failed"}:
                finished.append((job, state))
        if not finished:
            return

        for job, state in finished:
            self._running.remove(job)
            self.available_budget = self.available_budget.add(job.resources)
            self._release_allocations(job.assigned_cpu_indices, job.resources.gpu_runtime, job.assigned_gpu_indices)
            if state == "done":
                runtime_job_done(id(job))
            elif state == "failed":
                runtime_job_failed(id(job))

        self._condition.notify_all()

    def _start_ready_jobs_locked(self) -> bool:
        """Start pending jobs that are dependency-ready and resource-feasible.

        Returns:
            ``True`` if at least one pending job changed state.
        """
        started_any = False

        for job in list(self._pending):
            if not self._deps_ready(job):
                # If deps failed, _deps_ready() marks failed and removes from pending.
                started_any = started_any or (job not in self._pending)
                continue

            if not self.available_budget.fits(job.resources):
                continue

            allocations = self._reserve_indices(job.resources)
            if allocations is None:
                continue
            cpu_indices, gpu_indices = allocations

            try:
                self._launch_job(job, cpu_indices=cpu_indices, gpu_indices=gpu_indices)
            except (OSError, RuntimeError, ValueError):
                self._mark_pending_failed(job, cpu_indices=cpu_indices, gpu_indices=gpu_indices)
                started_any = True
                continue

            self.available_budget = self.available_budget.subtract(job.resources)
            self._pending.remove(job)
            self._running.add(job)
            started_any = True

        return started_any

    def _deps_ready(self, job: LocalJob) -> bool:
        """Return whether all job dependencies are complete and successful."""
        states = {dep.state() for dep in job.dependencies}
        if "failed" in states:
            self._mark_pending_failed(job)
            return False
        return not states or states == {"done"}

    def _launch_job(self, job: LocalJob, cpu_indices: list[int], gpu_indices: list[int]) -> None:
        """Spawn the child process for one job and bind resource metadata."""
        assigned_resources = AssignedResources(
            cpu_indices=cpu_indices,
            gpu_indices=gpu_indices,
            memory=job.resources.memory,
            gpu_memory=job.resources.gpu_memory,
        )

        job.job_id, argv, env_overrides = job.snapshot.prepare_job(
            work_unit=job.work_unit,
            workspace=job.workspace,
            assigned_resources_getter=(lambda r=assigned_resources: r),
        )

        assigned_resources_env = build_assigned_resources_env(
            assigned_resources=assigned_resources, gpu_runtime=job.resources.gpu_runtime, source="inline"
        )

        job.log_path = job.workspace.get_job_log(job_id=job.job_id, work_unit=job.work_unit)
        log_fp = job.log_path.open("ab", buffering=0)
        try:
            process = subprocess.Popen(  # noqa: S603
                argv,
                env=os.environ | env_overrides | assigned_resources_env,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            # Don't leak a file descriptor if Popen fails.
            log_fp.close()
            raise

        job.set_process(process, log_fp=log_fp)
        job.assigned_cpu_indices = cpu_indices
        job.assigned_gpu_indices = gpu_indices
        runtime_job_running(id(job), job_id=job.job_id, pid=process.pid)

    def _reserve_indices(self, resources: Resources) -> tuple[list[int], list[int]] | None:
        """Reserve CPU/GPU indices for a resource request."""
        cpu_indices = self._allocate_indices(self.available_cpu_indices, resources.cpus)
        if cpu_indices is None:
            return None

        gpu_pool = self._gpu_index_pool(resources.gpu_runtime)
        gpu_indices = self._allocate_indices(gpu_pool, resources.gpus)
        if gpu_indices is None:
            self._release_allocations(cpu_indices, resources.gpu_runtime, [])
            return None
        return cpu_indices, gpu_indices

    def _release_allocations(self, cpu_indices: list[int], gpu_runtime: GpuRuntime, gpu_indices: list[int]) -> None:
        """Return previously reserved CPU/GPU indices to the free pools."""
        self._release_indices(self.available_cpu_indices, cpu_indices)
        self._release_indices(self._gpu_index_pool(gpu_runtime), gpu_indices)

    def _mark_pending_failed(
        self,
        job: LocalJob,
        cpu_indices: list[int] | None = None,
        gpu_indices: list[int] | None = None,
    ) -> None:
        """Mark a pending job failed and release any provisional allocations."""
        if cpu_indices is None:
            cpu_indices = []
        if gpu_indices is None:
            gpu_indices = []
        self._release_allocations(cpu_indices, job.resources.gpu_runtime, gpu_indices)
        job.mark_failed()
        if job in self._pending:
            self._pending.remove(job)
        runtime_job_failed(id(job))

    def _gpu_index_pool(self, gpu_runtime: GpuRuntime) -> list[int]:
        match gpu_runtime:
            case "cuda":
                return self.available_cuda_gpu_indices
            case "rocm":
                return self.available_rocm_gpu_indices
            case "xpu":
                return self.available_xpu_gpu_indices

    @staticmethod
    def _allocate_indices(pool: list[int], count: int) -> list[int] | None:
        """Allocate `count` indices from a sorted index pool."""
        if count == 0:
            return []
        if len(pool) < count:
            return None
        return pool[:count]

    @staticmethod
    def _release_indices(pool: list[int], indices: list[int]) -> None:
        """Return indices to a pool while preserving sorted order."""
        for idx in indices:
            insort(pool, idx)


class LocalExecutor(Executor[LocalJob, LocalSnapshot]):
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

    def __post_init__(self) -> None:
        """Infer resource limits and initialize scheduler."""
        if self.max_memory == "all":
            self.max_memory = _infer_total_memory_gb()

        cpu_indices = _resolve_cpu_indices(num_cpus=self.num_cpus, cpu_indices=self.cpu_indices)
        cuda_gpu_indices = _resolve_gpu_indices(
            gpu_runtime="cuda",
            num_gpus=self.num_cuda_gpus,
            gpu_indices=self.cuda_gpu_indices,
        )
        rocm_gpu_indices = _resolve_gpu_indices(
            gpu_runtime="rocm",
            num_gpus=self.num_rocm_gpus,
            gpu_indices=self.rocm_gpu_indices,
        )
        xpu_gpu_indices = _resolve_gpu_indices(
            gpu_runtime="xpu",
            num_gpus=self.num_xpu_gpus,
            gpu_indices=self.xpu_gpu_indices,
        )

        self._resource_budget = _ResourceBudget(
            memory=self.max_memory,
            cpus=len(cpu_indices),
            cuda_gpus=len(cuda_gpu_indices),
            rocm_gpus=len(rocm_gpu_indices),
            xpu_gpus=len(xpu_gpu_indices),
        )
        self._scheduler = _LocalScheduler(
            self._resource_budget,
            available_cpu_indices=cpu_indices,
            available_cuda_gpu_indices=cuda_gpu_indices,
            available_rocm_gpu_indices=rocm_gpu_indices,
            available_xpu_gpu_indices=xpu_gpu_indices,
        )

    @classmethod
    @cache
    def _cached_snapshot(cls, snapshots_dir: Path) -> LocalSnapshot:
        """Return cached local snapshot instance for a snapshots directory."""
        return LocalSnapshot(snapshots_dir=snapshots_dir)

    def _make_snapshot(self, workspace: Workspace) -> LocalSnapshot:
        """Create or reuse snapshot for this submit call."""
        snapshots_dir = (workspace.get_temp_dir() / "snapshots").resolve()
        return LocalExecutor._cached_snapshot(snapshots_dir=snapshots_dir)

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[LocalJob],
        workspace: Workspace,
        snapshot: LocalSnapshot,
    ) -> LocalJob:
        """Queue a work unit in the local scheduler."""
        resources = work_unit.resources
        if not self._resource_budget.fits(resources):
            msg = (
                "Requested resources exceed LocalExecutor limits: "
                f"requested cpus={resources.cpus}, memory={resources.memory}, "
                f"gpus={resources.gpus} (runtime={resources.gpu_runtime}); "
                f"limits cpus={self._resource_budget.cpus}, memory={self._resource_budget.memory}, "
                f"cuda_gpus={self._resource_budget.cuda_gpus}, rocm_gpus={self._resource_budget.rocm_gpus}, "
                f"xpu_gpus={self._resource_budget.xpu_gpus}."
            )
            raise ValueError(msg)

        job = LocalJob(
            work_unit=work_unit,
            resources=resources,
            dependencies=dependencies,
            snapshot=snapshot,
            workspace=workspace,
        )
        runtime_job_pending(id(job), label=work_unit_label(work_unit))
        self._scheduler.submit(job)
        return job


def _infer_total_memory_gb() -> int:
    """Infer total physical memory in GiB.

    Returns:
        Inferred memory, with a conservative fallback of ``1``.
    """
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        total_bytes = int(page_size) * int(page_count)
        return max(1, total_bytes // (1024**3))
    except (ValueError, OSError, AttributeError):
        return 1


def _resolve_cpu_indices(num_cpus: int | Literal["all"], cpu_indices: list[int] | None) -> list[int]:
    """Resolve scheduler CPU pool from count/list configuration."""
    if cpu_indices is not None:
        normalized_cpu_indices = _normalize_indices(cpu_indices, name="cpu_indices")
        if num_cpus != "all" and num_cpus != len(normalized_cpu_indices):
            msg = (
                f"num_cpus={num_cpus} does not match len(cpu_indices)={len(normalized_cpu_indices)}. "
                "Set num_cpus='all' when using explicit cpu_indices."
            )
            raise ValueError(msg)
        return normalized_cpu_indices

    if num_cpus == "all":
        inferred_cpus = os.cpu_count() or 1
        return list(range(inferred_cpus))
    if num_cpus < 1:
        msg = "num_cpus must be >= 1."
        raise ValueError(msg)
    return list(range(num_cpus))


def _resolve_gpu_indices(gpu_runtime: GpuRuntime, num_gpus: int, gpu_indices: list[int] | None) -> list[int]:
    """Resolve scheduler GPU pool for one runtime from count/list configuration."""
    if num_gpus < 0:
        msg = f"num_{gpu_runtime}_gpus must be >= 0."
        raise ValueError(msg)

    if gpu_indices is not None:
        normalized_gpu_indices = _normalize_indices(gpu_indices, name=f"{gpu_runtime}_gpu_indices")
        if num_gpus != 0 and num_gpus != len(normalized_gpu_indices):
            msg = (
                f"num_{gpu_runtime}_gpus={num_gpus} does not match "
                f"len({gpu_runtime}_gpu_indices)={len(normalized_gpu_indices)}. "
                f"Set num_{gpu_runtime}_gpus=0 when using explicit {gpu_runtime}_gpu_indices."
            )
            raise ValueError(msg)
        return normalized_gpu_indices

    return list(range(num_gpus))


def _normalize_indices(indices: list[int], name: str) -> list[int]:
    if any(index < 0 for index in indices):
        msg = f"{name} must contain non-negative indices."
        raise ValueError(msg)
    if len(set(indices)) != len(indices):
        msg = f"{name} must not contain duplicates."
        raise ValueError(msg)
    return sorted(indices)
