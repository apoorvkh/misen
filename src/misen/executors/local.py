"""Local multiprocessing executor implementation."""

from __future__ import annotations

import multiprocessing
import os
import socket
import subprocess
import threading
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Literal

from misen.executor import Executor, Job
from misen.utils.snapshot import LocalSnapshot

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess
    from pathlib import Path

    from misen.task import Resources
    from misen.utils.assigned_resources import AssignedResources
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace


@dataclass(frozen=True)
class _ResourceBudget:
    """Simple resource budget tracker for local execution."""

    cpus: int
    memory: int
    gpus: int

    def fits(self, resources: Resources) -> bool:
        """Return True if the resources fit within the budget."""
        return resources.cpus <= self.cpus and resources.memory <= self.memory and resources.gpus <= self.gpus

    def subtract(self, resources: Resources) -> _ResourceBudget:
        """Return a new budget with the resources subtracted."""
        return _ResourceBudget(
            cpus=self.cpus - resources.cpus,
            memory=self.memory - resources.memory,
            gpus=self.gpus - resources.gpus,
        )

    def add(self, resources: Resources) -> _ResourceBudget:
        """Return a new budget with the resources added."""
        return _ResourceBudget(
            cpus=self.cpus + resources.cpus,
            memory=self.memory + resources.memory,
            gpus=self.gpus + resources.gpus,
        )


def _run_subprocess(argv: list[str], env: dict[str, str], log_path: Path) -> None:
    """Run a command in a worker process, writing output directly to the job log."""
    with log_path.open("a", buffering=1, encoding="utf-8", errors="replace") as log_file:
        try:
            completed = subprocess.run(  # noqa: S603
                argv,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        except FileNotFoundError:
            raise SystemExit(127) from None
    raise SystemExit(completed.returncode)


class LocalJob(Job):
    """Job implementation backed by a multiprocessing.Process."""

    __slots__ = (
        "_lock",
        "_process",
        "_state",
        "argv",
        "assigned_cpu_indices",
        "assigned_gpu_indices",
        "base_env",
        "dependencies",
        "env",
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
        base_env: dict[str, str],
    ) -> None:
        """Initialize a local job for a work unit."""
        super().__init__(work_unit=work_unit, job_id=None, log_path=None)
        self.resources = resources
        self.dependencies = dependencies
        self.argv: list[str] = []
        self.env = base_env
        self.base_env = base_env
        self.snapshot = snapshot
        self.workspace = workspace
        self.assigned_cpu_indices: tuple[int, ...] = ()
        self.assigned_gpu_indices: tuple[int, ...] = ()
        self._process: BaseProcess | None = None
        self._state: Literal["pending", "running", "done", "failed", "unknown"] = "pending"
        self._lock = threading.Lock()

    def state(self) -> Literal["pending", "running", "done", "failed", "unknown"]:
        """Return the current process state."""
        with self._lock:
            if self._state in {"done", "failed"}:
                return self._state

            if self._process is None:
                return "pending"

            if self._process.is_alive():
                return "running"

            if self._process.exitcode == 0:
                self._state = "done"
            elif self._process.exitcode is not None:
                self._state = "failed"
            return self._state

    def set_process(self, process: BaseProcess) -> None:
        """Set the underlying process and mark running."""
        with self._lock:
            self._process = process
            self._state = "running"

    def mark_failed(self) -> None:
        """Mark the job as failed."""
        with self._lock:
            self._state = "failed"


class _LocalScheduler:
    """Background scheduler for managing local job execution."""

    def __init__(self, total_budget: _ResourceBudget) -> None:
        """Initialize the scheduler with a total resource budget."""
        self._available_budget = total_budget
        self._available_cpu_indices = list(range(total_budget.cpus))
        self._available_gpu_indices = list(range(total_budget.gpus))
        hostname = socket.gethostname()
        self._hostnames = (hostname,) if hostname else ()
        self._pending: list[LocalJob] = []
        self._running: set[LocalJob] = set()
        self._condition = threading.Condition()
        self._context = multiprocessing.get_context("spawn")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, job: LocalJob) -> None:
        """Submit a job for scheduling."""
        with self._condition:
            self._pending.append(job)
            self._condition.notify_all()

    def _run(self) -> None:
        """Scheduler loop that launches ready jobs."""
        while True:
            with self._condition:
                self._collect_finished()
                started = self._start_ready_jobs()
                if not self._pending and not self._running:
                    self._condition.wait()
                elif not started:
                    self._condition.wait(timeout=0.1)

    def _collect_finished(self) -> None:
        """Collect finished jobs and release their resources."""
        finished = {job for job in self._running if job.state() in {"done", "failed"}}
        if not finished:
            return
        for job in finished:
            self._running.remove(job)
            self._available_budget = self._available_budget.add(job.resources)
            self._release_allocations(job.assigned_cpu_indices, job.assigned_gpu_indices)
        self._condition.notify_all()

    def _start_ready_jobs(self) -> bool:
        """Start pending jobs that are ready and return True if any started."""
        started_any = False
        for job in list(self._pending):
            dependency_states = {dep.state() for dep in job.dependencies}
            if "failed" in dependency_states:
                self._mark_pending_failed(job)
                started_any = True
                continue
            if dependency_states and dependency_states != {"done"}:
                continue

            resources = job.resources
            if not self._available_budget.fits(resources):
                continue

            allocations = self._reserve_indices(resources)
            if allocations is None:
                continue
            cpu_indices, gpu_indices = allocations

            assigned_resources: AssignedResources = {
                "executor": "local",
                "hostnames": self._hostnames,
                "cpu_indices": cpu_indices,
                "gpu_indices": gpu_indices,
                "cpu_count": resources.cpus,
                "gpu_count": resources.gpus,
            }
            try:
                job_id, argv, env_overrides = job.snapshot.prepare_job(
                    work_unit=job.work_unit,
                    workspace=job.workspace,
                    assigned_resources=assigned_resources,
                )
                job.job_id = job_id
                job.argv = argv
                job.env = os.environ | env_overrides
                job.log_path = job.workspace.get_job_log(job_id=job_id, work_unit=job.work_unit)
            except (OSError, RuntimeError, ValueError):
                self._mark_pending_failed(job, cpu_indices=cpu_indices, gpu_indices=gpu_indices)
                started_any = True
                continue

            process = self._context.Process(
                target=_run_subprocess,
                args=(job.argv, job.env, job.log_path),
            )
            try:
                process.start()
            except (OSError, RuntimeError):
                self._mark_pending_failed(job, cpu_indices=cpu_indices, gpu_indices=gpu_indices)
                started_any = True
                continue

            job.set_process(process)
            job.assigned_cpu_indices = cpu_indices
            job.assigned_gpu_indices = gpu_indices
            self._available_budget = self._available_budget.subtract(resources)
            self._pending.remove(job)
            self._running.add(job)
            started_any = True
        return started_any

    def _reserve_indices(self, resources: Resources) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        cpu_indices = self._allocate_indices(self._available_cpu_indices, resources.cpus)
        if cpu_indices is None:
            return None
        gpu_indices = self._allocate_indices(self._available_gpu_indices, resources.gpus)
        if gpu_indices is None:
            self._release_allocations(cpu_indices, ())
            return None
        return cpu_indices, gpu_indices

    def _release_allocations(self, cpu_indices: tuple[int, ...], gpu_indices: tuple[int, ...]) -> None:
        self._release_indices(self._available_cpu_indices, cpu_indices)
        self._release_indices(self._available_gpu_indices, gpu_indices)

    def _mark_pending_failed(
        self, job: LocalJob, *, cpu_indices: tuple[int, ...] = (), gpu_indices: tuple[int, ...] = ()
    ) -> None:
        self._release_allocations(cpu_indices, gpu_indices)
        job.mark_failed()
        self._pending.remove(job)

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
        if not indices:
            return
        pool.extend(indices)
        pool.sort()


class LocalExecutor(Executor[LocalJob, LocalSnapshot]):
    """Executor implementation that runs tasks locally."""

    max_cpus: int | None = None
    max_memory: int | None = None
    max_gpus: int | None = None

    def __post_init__(self) -> None:
        """Initialize the local scheduler and resource budget."""
        self._resource_budget = _ResourceBudget(
            cpus=self.max_cpus or (os.cpu_count() or 1),
            memory=self.max_memory or _infer_total_memory_gb(),
            gpus=self.max_gpus or 0,
        )
        self._scheduler = _LocalScheduler(self._resource_budget)

    @classmethod
    @cache
    def _cached_snapshot(cls, snapshots_dir: Path) -> LocalSnapshot:
        """Return a cached local snapshot instance for this executor class."""
        return LocalSnapshot(snapshots_dir=snapshots_dir)

    def _make_snapshot(self, workspace: Workspace) -> LocalSnapshot:
        """Create or reuse the local executor snapshot."""
        snapshots_dir = (workspace.get_temp_dir() / "snapshots").resolve()
        return LocalExecutor._cached_snapshot(snapshots_dir=snapshots_dir)

    def _dispatch(
        self, work_unit: WorkUnit, dependencies: set[LocalJob], workspace: Workspace, snapshot: LocalSnapshot
    ) -> LocalJob:
        """Dispatch a work unit to the local scheduler."""
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
            base_env=os.environ.copy(),
        )
        self._scheduler.submit(job)
        return job


def _infer_total_memory_gb() -> int:
    """Infer total system memory in gigabytes."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        total_bytes = int(page_size) * int(page_count)
        return max(1, total_bytes // (1024**3))
    except (ValueError, OSError, AttributeError):
        return 1
