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


def _infer_total_memory_gb() -> int:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        total_bytes = int(page_size) * int(page_count)
        return max(1, total_bytes // (1024**3))
    except (ValueError, OSError, AttributeError):
        return 1


@dataclass(frozen=True)
class _ResourceBudget:
    cpus: int
    memory: int
    gpus: int

    def fits(self, resources: TaskResources) -> bool:
        return resources.cpus <= self.cpus and resources.memory <= self.memory and resources.gpus <= self.gpus

    def subtract(self, resources: TaskResources) -> _ResourceBudget:
        return _ResourceBudget(
            cpus=self.cpus - resources.cpus,
            memory=self.memory - resources.memory,
            gpus=self.gpus - resources.gpus,
        )

    def add(self, resources: TaskResources) -> _ResourceBudget:
        return _ResourceBudget(
            cpus=self.cpus + resources.cpus,
            memory=self.memory + resources.memory,
            gpus=self.gpus + resources.gpus,
        )


def _run_pickled(payload: bytes) -> None:
    func = cloudpickle.loads(payload)
    func()


class LocalJob(Job):
    def __init__(self, resources: TaskResources, dependencies: set["LocalJob"], payload: bytes):
        self.resources = resources
        self.dependencies = dependencies
        self.payload = payload
        self._process: multiprocessing.Process | None = None
        self._state: Literal["pending", "running", "done", "failed", "unknown"] = "pending"
        self._lock = threading.Lock()

    def state(self) -> Literal["pending", "running", "done", "failed", "unknown"]:
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
        with self._lock:
            self._process = process
            self._state = "running"

    def _mark_failed(self) -> None:
        with self._lock:
            self._state = "failed"


class _LocalScheduler:
    def __init__(self, total_budget: _ResourceBudget):
        self._total_budget = total_budget
        self._available_budget = total_budget
        self._pending: list[LocalJob] = []
        self._running: set[LocalJob] = set()
        self._condition = threading.Condition()
        self._context = multiprocessing.get_context("spawn")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, job: LocalJob) -> None:
        with self._condition:
            self._pending.append(job)
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._collect_finished()
                started = self._start_ready_jobs()
                if not self._pending and not self._running:
                    self._condition.wait()
                elif not started:
                    self._condition.wait(timeout=0.1)

    def _collect_finished(self) -> None:
        finished = {job for job in self._running if job.state() in {"done", "failed"}}
        if not finished:
            return
        for job in finished:
            self._running.remove(job)
            self._available_budget = self._available_budget.add(job.resources)
        self._condition.notify_all()

    def _start_ready_jobs(self) -> bool:
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
    max_cpus: int | None = None
    max_memory: int | None = None
    max_gpus: int | None = None

    def __post_init__(self) -> None:
        self._resource_budget = _ResourceBudget(
            cpus=self.max_cpus or (os.cpu_count() or 1),
            memory=self.max_memory or _infer_total_memory_gb(),
            gpus=self.max_gpus or 0,
        )
        self._scheduler = _LocalScheduler(self._resource_budget)

    def _dispatch(self, work_unit: WorkUnit, dependencies: set[LocalJob], workspace: Workspace) -> LocalJob:
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
        job = LocalJob(resources=resources, dependencies=dependencies, payload=payload)
        self._scheduler.submit(job)
        return job
