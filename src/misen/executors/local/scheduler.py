"""Background scheduling loop for local executor jobs."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from bisect import insort
from typing import TYPE_CHECKING

import msgspec

from misen.utils.assigned_resources import AssignedResources
from misen.utils.runtime_events import runtime_job_done, runtime_job_failed, runtime_job_running

if TYPE_CHECKING:
    from misen.executors.local.budget import ResourceBudget
    from misen.executors.local.executor import JobState, LocalJob
    from misen.task_metadata import GpuRuntime, Resources


class LocalScheduler(msgspec.Struct, dict=True):
    """Background scheduler for local jobs.

    The scheduler owns:

    - pending/running queues
    - resource allocation state
    - dependency readiness checks

    It is intentionally isolated from :class:`LocalExecutor` so executor
    dispatch stays lightweight and thread-safe.
    """

    available_budget: ResourceBudget
    available_cpu_indices: list[int]
    available_cuda_gpu_indices: list[int]
    available_rocm_gpu_indices: list[int]
    available_xpu_gpu_indices: list[int]
    _logger = logging.getLogger(__name__)

    def __post_init__(self) -> None:
        """Initialize the scheduler."""
        self._pending: list[LocalJob] = []
        self._running: set[LocalJob] = set()
        self._condition = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
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
                self._logger.info("LocalScheduler observed job completion for %s.", job.work_unit)
                runtime_job_done(id(job))
            elif state == "failed":
                self._logger.error("LocalScheduler observed job failure for %s.", job.work_unit)
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
            self._logger.error("Dependency failed; marking pending job failed for %s.", job.work_unit)
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
            gpu_runtime=job.resources.gpu_runtime,
        )

        job.log_path = job.workspace.get_job_log(job_id=job.job_id, work_unit=job.work_unit)
        log_fp = job.log_path.open("ab", buffering=0)
        self._logger.debug(
            "Launching local subprocess for %s with job_id=%s cpu_indices=%s gpu_indices=%s log=%s.",
            job.work_unit,
            job.job_id,
            cpu_indices,
            gpu_indices,
            job.log_path,
        )
        try:
            process = subprocess.Popen(  # noqa: S603
                argv,
                env=os.environ | env_overrides,
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
        self._logger.info(
            "Launched local subprocess for %s (job_id=%s, pid=%d).",
            job.work_unit,
            job.job_id,
            process.pid,
        )
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
        self._logger.error("Marked pending local job failed for %s.", job.work_unit)
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
        allocated = pool[:count]
        del pool[:count]
        return allocated

    @staticmethod
    def _release_indices(pool: list[int], indices: list[int]) -> None:
        """Return indices to a pool while preserving sorted order."""
        for idx in indices:
            insort(pool, idx)
