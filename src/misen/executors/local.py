"""Local subprocess-based executor implementation.

Concurrency model:

- Submit-time builds a dependency graph of work units.
- A background scheduler thread launches ready jobs when resources fit.
- Resource budgeting is explicit (CPUs, memory, GPUs) and prevents oversubscription.
- Dependency failures propagate to downstream pending jobs.

Execution still uses the generic executor contract (snapshot + dispatch + job
handles), but this module performs all scheduling in-process.
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
from bisect import insort
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Literal

from misen.executor import Executor, Job
from misen.utils.assigned_resources import AssignedResources
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

    from misen.task_properties import Resources
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

JobState = Literal["pending", "running", "done", "failed", "unknown"]


@dataclass(frozen=True)
class _ResourceBudget:
    """Available resource budget for local scheduling.

    Values are interpreted as hard scheduler limits for concurrent job
    placement.
    """

    cpus: int
    memory: int
    gpus: int

    def fits(self, resources: Resources) -> bool:
        """Return whether requested resources fit in current budget.

        Args:
            resources: Requested work-unit resources.

        Returns:
            ``True`` if all requested dimensions are within budget.
        """
        return resources.cpus <= self.cpus and resources.memory <= self.memory and resources.gpus <= self.gpus

    def subtract(self, resources: Resources) -> _ResourceBudget:
        """Return new budget after reserving resources.

        Args:
            resources: Resources to reserve.

        Returns:
            Updated budget with resources subtracted.
        """
        return _ResourceBudget(
            cpus=self.cpus - resources.cpus,
            memory=self.memory - resources.memory,
            gpus=self.gpus - resources.gpus,
        )

    def add(self, resources: Resources) -> _ResourceBudget:
        """Return new budget after releasing resources.

        Args:
            resources: Resources to release.

        Returns:
            Updated budget with resources added.
        """
        return _ResourceBudget(
            cpus=self.cpus + resources.cpus,
            memory=self.memory + resources.memory,
            gpus=self.gpus + resources.gpus,
        )


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
        """Initialize a local job.

        Args:
            work_unit: Work unit to run.
            resources: Resource request for this work unit.
            dependencies: Upstream local jobs that must complete first.
            snapshot: Snapshot used to prepare execution command/env.
            workspace: Workspace for logs/artifacts.
        """
        super().__init__(work_unit=work_unit, job_id=None, log_path=None)
        self.resources = resources
        self.dependencies = dependencies
        self.snapshot = snapshot
        self.workspace = workspace

        self.assigned_cpu_indices: tuple[int, ...] = ()
        self.assigned_gpu_indices: tuple[int, ...] = ()

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
        """Attach process/log handles and mark job as running.

        Args:
            process: Spawned child process.
            log_fp: Open file descriptor receiving combined stdout/stderr.
        """
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


class _LocalScheduler:
    """Background scheduler for local jobs.

    The scheduler owns:

    - pending/running queues
    - resource allocation state
    - dependency readiness checks

    It is intentionally isolated from :class:`LocalExecutor` so executor
    dispatch stays lightweight and thread-safe.
    """

    def __init__(self, total_budget: _ResourceBudget) -> None:
        """Start scheduler loop with a fixed global resource budget.

        Args:
            total_budget: Maximum resources available for all concurrent jobs.
        """
        self._available_budget = total_budget
        self._available_cpu_indices = list(range(total_budget.cpus))
        self._available_gpu_indices = list(range(total_budget.gpus))

        hostname = socket.gethostname()
        self._hostnames = (hostname,) if hostname else ()

        self._pending: list[LocalJob] = []
        self._running: set[LocalJob] = set()

        self._condition = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, job: LocalJob) -> None:
        """Queue a job for scheduling.

        Args:
            job: Local job to enqueue.
        """
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
            self._available_budget = self._available_budget.add(job.resources)
            self._release_allocations(job.assigned_cpu_indices, job.assigned_gpu_indices)
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

            if not self._available_budget.fits(job.resources):
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

            self._available_budget = self._available_budget.subtract(job.resources)
            self._pending.remove(job)
            self._running.add(job)
            started_any = True

        return started_any

    def _deps_ready(self, job: LocalJob) -> bool:
        """Return whether all job dependencies are complete and successful.

        Args:
            job: Candidate pending job.

        Returns:
            ``True`` when dependencies are done; ``False`` otherwise.
        """
        states = {dep.state() for dep in job.dependencies}
        if "failed" in states:
            self._mark_pending_failed(job)
            return False
        return not states or states == {"done"}

    def _launch_job(self, job: LocalJob, cpu_indices: tuple[int, ...], gpu_indices: tuple[int, ...]) -> None:
        """Spawn the child process for one job and bind resource metadata.

        Args:
            job: Job to launch.
            cpu_indices: Physical CPU indices reserved for this job.
            gpu_indices: Physical GPU indices reserved for this job.
        """
        assigned_resources = AssignedResources(
            hostnames=self._hostnames,
            cpu_indices=cpu_indices,
            gpu_indices=gpu_indices,
            cpu_count=job.resources.cpus,
            gpu_count=job.resources.gpus,
        )

        job.job_id, argv, env_overrides = job.snapshot.prepare_job(
            work_unit=job.work_unit,
            workspace=job.workspace,
            assigned_resources_getter=(lambda r=assigned_resources: r),
        )

        job.log_path = job.workspace.get_job_log(job_id=job.job_id, work_unit=job.work_unit)
        log_fp = job.log_path.open("ab", buffering=0)
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
        runtime_job_running(id(job), job_id=job.job_id, pid=process.pid)

    def _reserve_indices(self, resources: Resources) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        """Reserve CPU/GPU indices for a resource request.

        Args:
            resources: Requested resources.

        Returns:
            Tuple of reserved CPU/GPU indices, or ``None`` if unavailable.
        """
        cpu_indices = self._allocate_indices(self._available_cpu_indices, resources.cpus)
        if cpu_indices is None:
            return None
        gpu_indices = self._allocate_indices(self._available_gpu_indices, resources.gpus)
        if gpu_indices is None:
            self._release_allocations(cpu_indices, ())
            return None
        return cpu_indices, gpu_indices

    def _release_allocations(self, cpu_indices: tuple[int, ...], gpu_indices: tuple[int, ...]) -> None:
        """Return previously reserved CPU/GPU indices to the free pools."""
        self._release_indices(self._available_cpu_indices, cpu_indices)
        self._release_indices(self._available_gpu_indices, gpu_indices)

    def _mark_pending_failed(
        self,
        job: LocalJob,
        cpu_indices: tuple[int, ...] = (),
        gpu_indices: tuple[int, ...] = (),
    ) -> None:
        """Mark a pending job failed and release any provisional allocations.

        Args:
            job: Job to fail.
            cpu_indices: CPU indices to release.
            gpu_indices: GPU indices to release.
        """
        self._release_allocations(cpu_indices, gpu_indices)
        job.mark_failed()
        if job in self._pending:
            self._pending.remove(job)
        runtime_job_failed(id(job))

    @staticmethod
    def _allocate_indices(pool: list[int], count: int) -> tuple[int, ...] | None:
        """Allocate `count` indices from a sorted index pool."""
        if count == 0:
            return ()
        if len(pool) < count:
            return None
        indices = tuple(pool[:count])
        del pool[:count]
        return indices

    @staticmethod
    def _release_indices(pool: list[int], indices: tuple[int, ...]) -> None:
        """Return indices to a pool while preserving sorted order."""
        for idx in indices:
            insort(pool, idx)


class LocalExecutor(Executor[LocalJob, LocalSnapshot]):
    """Executor that runs work units as local subprocesses."""

    max_cpus: int | None = None
    max_memory: int | None = None
    max_gpus: int | None = None

    def __post_init__(self) -> None:
        """Initialize scheduler with configured or inferred resource limits."""
        self._resource_budget = _ResourceBudget(
            cpus=self.max_cpus or (os.cpu_count() or 1),
            memory=self.max_memory or _infer_total_memory_gb(),
            gpus=self.max_gpus or 0,
        )
        self._scheduler = _LocalScheduler(self._resource_budget)

    @classmethod
    @cache
    def _cached_snapshot(cls, snapshots_dir: Path) -> LocalSnapshot:
        """Return cached local snapshot instance for a snapshots directory."""
        return LocalSnapshot(snapshots_dir=snapshots_dir)

    def _make_snapshot(self, workspace: Workspace) -> LocalSnapshot:
        """Create or reuse snapshot for this submit call.

        Args:
            workspace: Workspace used to locate snapshot directory.

        Returns:
            Snapshot instance.
        """
        snapshots_dir = (workspace.get_temp_dir() / "snapshots").resolve()
        return LocalExecutor._cached_snapshot(snapshots_dir=snapshots_dir)

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[LocalJob],
        workspace: Workspace,
        snapshot: LocalSnapshot,
    ) -> LocalJob:
        """Queue a work unit in the local scheduler.

        Args:
            work_unit: Work unit to execute.
            dependencies: Upstream jobs that must complete first.
            workspace: Workspace for logs/artifacts.
            snapshot: Snapshot used to generate payload command.

        Returns:
            Job handle representing scheduled work.

        Raises:
            ValueError: If requested resources exceed configured local limits.
        """
        resources = work_unit.resources
        if not self._resource_budget.fits(resources):
            msg = (
                "Requested resources exceed LocalExecutor limits: "
                f"requested cpus={resources.cpus}, memory={resources.memory}, gpus={resources.gpus}; "
                f"limits cpus={self._resource_budget.cpus}, memory={self._resource_budget.memory}, "
                f"gpus={self._resource_budget.gpus}."
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
