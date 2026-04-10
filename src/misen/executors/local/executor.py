"""Local subprocess-based executor implementation."""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Literal

from misen.executor import Executor, Job
from misen.executors.local.budget import ResourceBudget
from misen.executors.local.scheduler import LocalScheduler
from misen.utils.runtime_events import runtime_job_pending, work_unit_label
from misen.utils.snapshot import LocalSnapshot

if TYPE_CHECKING:
    import subprocess
    from io import FileIO

    from misen.task_metadata import GpuRuntime, Resources
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace


JobState = Literal["pending", "running", "done", "failed", "unknown"]
logger = logging.getLogger(__name__)


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
            logger.info(
                "Local job %s for %s exited with code %d -> state=%s.",
                self.job_id or "n/a",
                work_unit_label(self.work_unit),
                rc,
                self._state,
            )
            return self._state

    def set_process(self, process: subprocess.Popen[bytes], log_fp: FileIO) -> None:
        """Attach process/log handles and mark job as running."""
        with self._lock:
            self._process = process
            self._log_fp = log_fp
            self._state = "running"
            logger.info(
                "Local job %s for %s is running (pid=%d).",
                self.job_id or "n/a",
                work_unit_label(self.work_unit),
                process.pid,
            )

    def mark_failed(self) -> None:
        """Mark pending/running job as failed and close log handles."""
        with self._lock:
            self._close_log_fp_locked()
            self._state = "failed"
            logger.error("Local job %s for %s marked failed.", self.job_id or "n/a", work_unit_label(self.work_unit))

    def _close_log_fp_locked(self) -> None:
        fp = self._log_fp
        if fp is None:
            return
        try:
            fp.close()
        finally:
            self._log_fp = None


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

        self._resource_budget = ResourceBudget(
            memory=self.max_memory,
            cpus=len(cpu_indices),
            cuda_gpus=len(cuda_gpu_indices),
            rocm_gpus=len(rocm_gpu_indices),
            xpu_gpus=len(xpu_gpu_indices),
        )
        logger.info(
            (
                "Initialized LocalExecutor budget: memory=%sGiB cpus=%d cuda_gpus=%d "
                "rocm_gpus=%d xpu_gpus=%d."
            ),
            self._resource_budget.memory,
            self._resource_budget.cpus,
            self._resource_budget.cuda_gpus,
            self._resource_budget.rocm_gpus,
            self._resource_budget.xpu_gpus,
        )
        self._scheduler = LocalScheduler(
            self._resource_budget,
            available_cpu_indices=cpu_indices,
            available_cuda_gpu_indices=cuda_gpu_indices,
            available_rocm_gpu_indices=rocm_gpu_indices,
            available_xpu_gpu_indices=xpu_gpu_indices,
        )

    def _make_snapshot(self, workspace: Workspace) -> LocalSnapshot:
        """Return cached ``LocalSnapshot`` rooted at workspace snapshots dir."""
        return self._make_local_snapshot(workspace=workspace)

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
        logger.debug(
            "Queued local work unit %s with resources=%s and %d dependency job(s).",
            work_unit_label(work_unit),
            resources,
            len(dependencies),
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
