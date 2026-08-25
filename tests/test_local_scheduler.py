"""Dependency and resource scheduling for the local executor."""
# ruff: noqa: D103, S101, SLF001

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

import misen.executors.local as local_module
from misen import Task, meta
from misen.exceptions import ExecutionError, StorageError
from misen.executors.local import LocalJob
from misen.utils.work_unit import WorkUnit

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from misen.task_metadata import Resources
    from misen.workspace import Workspace


@meta(id="local_scheduler_test")
def _scheduler_task(value: int) -> int:
    return value


def _scheduler(monkeypatch: pytest.MonkeyPatch, *, cpus: int = 2) -> local_module._LocalScheduler:
    monkeypatch.setattr(local_module.threading.Thread, "start", lambda _: None)
    monkeypatch.setattr(local_module.atexit, "register", lambda _: None)
    monkeypatch.setattr(local_module, "_install_sigterm_handler", lambda _: None)
    monkeypatch.setattr(local_module, "runtime_job_done", lambda _: None)
    monkeypatch.setattr(local_module, "runtime_job_failed", lambda _: None)
    return local_module._LocalScheduler(
        available_budget=local_module._ResourceBudget(
            memory=64,
            cpus=cpus,
            accelerators=0,
            accelerator_type="cuda",
        ),
        available_cpu_indices=list(range(cpus)),
        available_accelerator_indices=[],
    )


def _job(value: int, *, dependencies: Iterable[LocalJob] = (), cpus: int = 1) -> LocalJob:
    task = Task(_scheduler_task, value).with_resources(memory=1, cpus=cpus)
    return LocalJob(
        WorkUnit(root=task, dependencies=set()),
        dependencies=set(dependencies),
        snapshot=None,
        workspace=cast("Workspace", None),
    )


def _record_launches(
    scheduler: local_module._LocalScheduler,
    monkeypatch: pytest.MonkeyPatch,
) -> list[LocalJob]:
    launched: list[LocalJob] = []

    def launch(
        _: local_module._LocalScheduler,
        job: LocalJob,
        *,
        cpu_indices: list[int],
        accelerator_indices: list[int],
    ) -> None:
        job.assigned_cpu_indices = list(cpu_indices)
        job.assigned_accelerator_indices = list(accelerator_indices)
        job._cached_state = "running"
        launched.append(job)

    monkeypatch.setattr(type(scheduler), "_launch_job", launch)
    return launched


def test_successful_job_releases_its_dependents(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _scheduler(monkeypatch)
    launched = _record_launches(scheduler, monkeypatch)
    parent = _job(0)
    children = [_job(i, dependencies=[parent]) for i in (1, 2)]

    for job in (parent, *children):
        scheduler.submit(job)
    with scheduler._condition:
        scheduler._tick_locked()
        assert launched == [parent]
        parent._cached_state = "done"
        scheduler._tick_locked()

    assert launched == [parent, *children]
    assert not scheduler._waiting


def test_job_waits_for_every_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _scheduler(monkeypatch)
    launched = _record_launches(scheduler, monkeypatch)
    parents = [_job(0), _job(1)]
    child = _job(2, dependencies=parents)

    for job in (*parents, child):
        scheduler.submit(job)
    with scheduler._condition:
        scheduler._tick_locked()
        parents[0]._cached_state = "done"
        scheduler._tick_locked()
        assert launched == parents
        parents[1]._cached_state = "done"
        scheduler._tick_locked()

    assert launched == [*parents, child]


def test_simultaneous_completions_preserve_submission_order(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _scheduler(monkeypatch)
    launched = _record_launches(scheduler, monkeypatch)
    parents = [_job(0), _job(1)]
    children = [_job(i + 2, dependencies=[parent]) for i, parent in enumerate(parents)]

    for job in (*parents, *children):
        scheduler.submit(job)
    with scheduler._condition:
        scheduler._tick_locked()
        for parent in parents:
            parent._cached_state = "done"
        scheduler._tick_locked()

    assert launched == [*parents, *children]


def test_scheduler_backfills_while_larger_job_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _scheduler(monkeypatch)
    launched = _record_launches(scheduler, monkeypatch)
    first, large, small = _job(0), _job(1, cpus=2), _job(2)

    scheduler.submit(first)
    with scheduler._condition:
        scheduler._tick_locked()
    scheduler.submit(large)
    scheduler.submit(small)
    with scheduler._condition:
        scheduler._tick_locked()
        assert launched == [first, small]
        small._cached_state = "done"
        scheduler._tick_locked()
        assert launched == [first, small]
        first._cached_state = "done"
        scheduler._tick_locked()

    assert launched == [first, small, large]
    assert large.assigned_cpu_indices == [0, 1]


def test_failed_job_transitively_fails_dependents(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _scheduler(monkeypatch)
    launched = _record_launches(scheduler, monkeypatch)
    parent = _job(0)
    child = _job(1, dependencies=[parent])
    grandchild = _job(2, dependencies=[child])

    for job in (parent, child, grandchild):
        scheduler.submit(job)
    with scheduler._condition:
        scheduler._tick_locked()
        parent._cached_state = "failed"
        scheduler._tick_locked()

    assert launched == [parent]
    assert child.state() == grandchild.state() == "failed"
    assert not scheduler._waiting


def test_log_finalization_failure_only_fails_affected_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scheduler = _scheduler(monkeypatch)
    affected, sibling = _job(0), _job(1)
    workspace = MagicMock()
    workspace.finalize_job_log.side_effect = StorageError("object store unavailable")
    affected.workspace = cast("Workspace", workspace)
    affected.log_path = tmp_path / "affected.log"
    affected._cached_state = "done"
    affected.assigned_cpu_indices = [0]
    sibling._cached_state = "running"
    sibling_process = MagicMock()
    sibling_process.status.is_active = True
    sibling._process = sibling_process
    sibling.assigned_cpu_indices = [1]

    scheduler.available_budget = scheduler.available_budget.subtract(affected.resources).subtract(sibling.resources)
    scheduler.available_cpu_indices.clear()
    scheduler._running[affected] = None
    scheduler._running[sibling] = None

    with scheduler._condition:
        scheduler._collect_finished_locked()

    assert affected._cached_state == "failed"
    assert affected.failure.reason == "Finalizing the job log failed: object store unavailable"
    assert affected not in scheduler._running
    assert sibling in scheduler._running
    assert scheduler._fatal_error is None
    assert scheduler.available_budget.cpus == 1
    assert scheduler.available_cpu_indices == [0]
    workspace.finalize_job_log.assert_called_once_with(affected.log_path)


def test_shutdown_stops_every_job_after_one_stop_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _scheduler(monkeypatch)
    failed_stop = MagicMock(spec=LocalJob)
    sibling = MagicMock(spec=LocalJob)
    failed_stop.astop.side_effect = RuntimeError("stop failed")
    sibling.astop.return_value = True
    monkeypatch.setattr(local_module, "_SHUTDOWN_TERM_GRACE_S", 0.01)
    scheduler._running = {failed_stop: None, sibling: None}

    scheduler._terminate_running_jobs()

    reason = "Local executor shut down before the job completed."
    failed_stop.astop.assert_awaited_once_with(grace_seconds=0.01, reason=reason)
    sibling.astop.assert_awaited_once_with(grace_seconds=0.01, reason=reason)


def test_shutdown_fails_queued_jobs_and_rejects_new_submissions(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _scheduler(monkeypatch)
    queued = _job(0)
    scheduler.submit(queued)

    scheduler._terminate_running_jobs()

    assert queued.state() == "failed"
    assert queued.failure.reason == "Local executor shut down before the job completed."
    with pytest.raises(ExecutionError, match="shutting down"):
        scheduler.submit(_job(1))


def test_launch_failure_releases_resources_and_fails_dependents(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _scheduler(monkeypatch, cpus=1)
    parent = _job(0)
    child = _job(1, dependencies=[parent])

    def fail_launch(*_: object, **__: object) -> None:
        msg = "launch failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(type(scheduler), "_launch_job", fail_launch)
    scheduler.submit(parent)
    scheduler.submit(child)
    with scheduler._condition:
        scheduler._tick_locked()

    assert parent.state() == child.state() == "failed"
    assert parent.failure.reason == "Launch failed: RuntimeError: launch failed"
    assert child.failure.reason == "A dependency failed."
    assert scheduler.available_budget.cpus == 1
    assert scheduler.available_cpu_indices == [0]


def test_submit_handles_terminal_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _scheduler(monkeypatch)
    launched = _record_launches(scheduler, monkeypatch)
    done, failed = _job(0), _job(1)
    done._cached_state = "done"
    failed._cached_state = "failed"
    ready = _job(2, dependencies=[done])
    rejected = _job(3, dependencies=[failed])

    scheduler.submit(ready)
    scheduler.submit(rejected)
    with scheduler._condition:
        scheduler._tick_locked()

    assert launched == [ready]
    assert rejected.state() == "failed"


def test_waiting_dependencies_are_not_repolled(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _scheduler(monkeypatch, cpus=1)
    _record_launches(scheduler, monkeypatch)
    state_calls = 0
    original_state = LocalJob.state

    def count_state(job: LocalJob) -> local_module._JobState:
        nonlocal state_calls
        state_calls += 1
        return original_state(job)

    monkeypatch.setattr(LocalJob, "state", count_state)
    jobs: list[LocalJob] = []
    for i in range(2_000):
        jobs.append(_job(i, dependencies=jobs[-1:]))
        scheduler.submit(jobs[-1])

    with scheduler._condition:
        for _ in range(5):
            scheduler._start_ready_jobs_locked()

    assert state_calls <= len(jobs)


def test_resource_admission_is_linear_for_serial_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _scheduler(monkeypatch, cpus=1)
    launched = _record_launches(scheduler, monkeypatch)
    jobs = [_job(i) for i in range(1_000)]
    fit_calls = 0
    original_fits = local_module._ResourceBudget.fits

    def count_fits(budget: local_module._ResourceBudget, resources: Resources) -> bool:
        nonlocal fit_calls
        fit_calls += 1
        return original_fits(budget, resources)

    monkeypatch.setattr(local_module._ResourceBudget, "fits", count_fits)
    for job in jobs:
        scheduler.submit(job)
    with scheduler._condition:
        scheduler._tick_locked()
        for job in jobs:
            job._cached_state = "done"
            scheduler._tick_locked()

    assert launched == jobs
    assert fit_calls <= 2 * len(jobs)
