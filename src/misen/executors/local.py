from __future__ import annotations

import functools
import multiprocessing
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import cloudpickle

from misen.executor import Executor, Job, WorkUnit

if TYPE_CHECKING:
    from misen.task import TaskResources
    from misen.workspace import Workspace


@dataclass(frozen=True)
class _ResourceBudget:
    """Simple resource budget tracker for local execution."""

    cpus: int
    memory: int
    gpus: int

    def fits(self, resources: TaskResources) -> bool:
        """Return True if the resources fit within the budget."""
        return resources.cpus <= self.cpus and resources.memory <= self.memory and resources.gpus <= self.gpus

    def subtract(self, resources: TaskResources) -> _ResourceBudget:
        """Return a new budget with the resources subtracted."""
        return _ResourceBudget(
            cpus=self.cpus - resources.cpus,
            memory=self.memory - resources.memory,
            gpus=self.gpus - resources.gpus,
        )

    def add(self, resources: TaskResources) -> _ResourceBudget:
        """Return a new budget with the resources added."""
        return _ResourceBudget(
            cpus=self.cpus + resources.cpus,
            memory=self.memory + resources.memory,
            gpus=self.gpus + resources.gpus,
        )


def _run_pickled(payload: bytes) -> None:
    """Run a pickled callable payload in a worker process."""
    func = cloudpickle.loads(payload)
    func()


class LocalJob(Job):
    """Job implementation backed by a multiprocessing.Process."""

    def __init__(
        self,
        work_unit: WorkUnit,
        resources: TaskResources,
        dependencies: set[LocalJob],
        payload: bytes,
    ) -> None:
        """Initialize a local job for a work unit."""
        super().__init__(work_unit=work_unit)
        self.resources = resources
        self.dependencies = dependencies
        self.payload = payload
        self._process: multiprocessing.Process | None = None
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

    def _set_process(self, process: multiprocessing.Process) -> None:
        """Set the underlying process and mark running."""
        with self._lock:
            self._process = process
            self._state = "running"

    def _mark_failed(self) -> None:
        """Mark the job as failed."""
        with self._lock:
            self._state = "failed"


class _LocalScheduler:
    """Background scheduler for managing local job execution."""

    def __init__(self, total_budget: _ResourceBudget) -> None:
        """Initialize the scheduler with a total resource budget."""
        self._total_budget = total_budget
        self._available_budget = total_budget
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
        self._condition.notify_all()

    def _start_ready_jobs(self) -> bool:
        """Start pending jobs that are ready and return True if any started."""
        started_any = False
        for job in list(self._pending):
            dependency_states = {dep.state() for dep in job.dependencies}
            if "failed" in dependency_states:
                job._mark_failed()
                self._pending.remove(job)
                started_any = True
                continue
            if dependency_states and dependency_states != {"done"}:
                continue
            if not self._available_budget.fits(job.resources):
                continue
            process = self._context.Process(target=_run_pickled, args=(job.payload,))
            process.start()
            job._set_process(process)
            self._available_budget = self._available_budget.subtract(job.resources)
            self._pending.remove(job)
            self._running.add(job)
            started_any = True
        return started_any


class LocalExecutor(Executor[LocalJob]):
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

    def _dispatch(self, work_unit: WorkUnit, dependencies: set[LocalJob], workspace: Workspace) -> LocalJob:
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
        payload = cloudpickle.dumps(functools.partial(work_unit.execute, workspace=workspace))
        job = LocalJob(work_unit=work_unit, resources=resources, dependencies=dependencies, payload=payload)
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
