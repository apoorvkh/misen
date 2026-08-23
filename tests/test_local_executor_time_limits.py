"""Tests for LocalExecutor's ``enforce_time_limits`` feature."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pytest

from misen import Task, meta
from misen.exceptions import ExecutionError
from misen.executors.local import LocalExecutor, LocalJob
from misen.utils.work_unit import WorkUnit
from misen.workspaces.disk import DiskWorkspace

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@meta(id="time_limited_local_task", cache=False, resources={"time": 1})
def _time_limited_task() -> int:
    return 1


@meta(id="default_resources_local_task", cache=False)
def _default_resources_task() -> int:
    return 1


def _make_local_job(task: Callable[[], int], tmp_path: Path) -> LocalJob:
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    work_unit = WorkUnit(root=Task(task), dependencies=set())
    return LocalJob(
        work_unit=work_unit,
        dependencies=set(),
        snapshot=None,
        workspace=workspace,
    )


class _FakeProcess:
    """Minimal subprocess.Popen stand-in for terminate() observability."""

    def __init__(self, *, terminate_returncode: int | None = -15) -> None:
        self.pid = 99999
        self.terminate_calls = 0
        self.kill_calls = 0
        self.terminate_returncode = terminate_returncode
        self._returncode: int | None = None

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._returncode = self.terminate_returncode

    def kill(self) -> None:
        self.kill_calls += 1
        self._returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self._returncode is None:
            raise TimeoutError(timeout)
        return self._returncode


def test_local_executor_default_does_not_enforce_time_limits() -> None:
    executor = LocalExecutor()
    assert executor.enforce_time_limits is False
    assert executor._scheduler.enforce_time_limits is False  # noqa: SLF001


def test_local_executor_propagates_enforce_time_limits_to_scheduler() -> None:
    executor = LocalExecutor(enforce_time_limits=True)
    assert executor.enforce_time_limits is True
    assert executor._scheduler.enforce_time_limits is True  # noqa: SLF001


def test_time_limit_exceeded_applies_default_60min_limit_to_unannotated_task(tmp_path: Path) -> None:
    job = _make_local_job(_default_resources_task, tmp_path)
    job._cached_state = "running"  # noqa: SLF001
    job._started_at = time.monotonic() - 3601  # 60-minute default, 1s past  # noqa: SLF001
    assert job.time_limit_exceeded() is True


def test_time_limit_exceeded_returns_false_before_start(tmp_path: Path) -> None:
    job = _make_local_job(_time_limited_task, tmp_path)
    assert job.time_limit_exceeded() is False


def test_time_limit_exceeded_returns_false_within_limit(tmp_path: Path) -> None:
    job = _make_local_job(_time_limited_task, tmp_path)
    job._cached_state = "running"  # noqa: SLF001
    job._started_at = time.monotonic()  # noqa: SLF001
    assert job.time_limit_exceeded() is False


def test_time_limit_exceeded_returns_true_when_running_past_limit(tmp_path: Path) -> None:
    job = _make_local_job(_time_limited_task, tmp_path)
    job._cached_state = "running"  # noqa: SLF001
    job._started_at = time.monotonic() - 61  # 1-minute limit, 61s elapsed  # noqa: SLF001
    assert job.time_limit_exceeded() is True


def test_time_limit_exceeded_returns_false_in_terminal_state(tmp_path: Path) -> None:
    job = _make_local_job(_time_limited_task, tmp_path)
    job._cached_state = "done"  # noqa: SLF001
    job._started_at = time.monotonic() - 1_000_000  # noqa: SLF001
    assert job.time_limit_exceeded() is False


def _stage_running_job(executor: LocalExecutor, job: LocalJob, *, fake_process: Any) -> None:
    job._process = fake_process  # noqa: SLF001
    job._cached_state = "running"  # noqa: SLF001


def test_scheduler_terminates_running_job_past_time_limit(tmp_path: Path) -> None:
    executor = LocalExecutor(enforce_time_limits=True)
    job = _make_local_job(_time_limited_task, tmp_path)
    job._started_at = time.monotonic() - 61  # noqa: SLF001
    fake_process = _FakeProcess()
    _stage_running_job(executor, job, fake_process=fake_process)

    with executor._scheduler._condition:  # noqa: SLF001
        executor._scheduler._running[job] = None  # noqa: SLF001
        executor._scheduler._terminate_timed_out_locked()  # noqa: SLF001
        executor._scheduler._running.pop(job, None)  # noqa: SLF001

    assert fake_process.terminate_calls == 1
    assert job.failure.reason == "Exceeded the requested 1 minute time limit."


def test_scheduler_does_not_terminate_running_job_within_time_limit(tmp_path: Path) -> None:
    executor = LocalExecutor(enforce_time_limits=True)
    job = _make_local_job(_time_limited_task, tmp_path)
    job._started_at = time.monotonic()  # noqa: SLF001
    fake_process = _FakeProcess()
    _stage_running_job(executor, job, fake_process=fake_process)

    with executor._scheduler._condition:  # noqa: SLF001
        executor._scheduler._running[job] = None  # noqa: SLF001
        executor._scheduler._terminate_timed_out_locked()  # noqa: SLF001
        executor._scheduler._running.pop(job, None)  # noqa: SLF001

    assert fake_process.terminate_calls == 0


def test_scheduler_terminates_unannotated_task_past_default_60min_limit(tmp_path: Path) -> None:
    executor = LocalExecutor(enforce_time_limits=True)
    job = _make_local_job(_default_resources_task, tmp_path)
    job._started_at = time.monotonic() - 3601  # 60-minute default, 1s past  # noqa: SLF001
    fake_process = _FakeProcess()
    _stage_running_job(executor, job, fake_process=fake_process)

    with executor._scheduler._condition:  # noqa: SLF001
        executor._scheduler._running[job] = None  # noqa: SLF001
        executor._scheduler._terminate_timed_out_locked()  # noqa: SLF001
        executor._scheduler._running.pop(job, None)  # noqa: SLF001

    assert fake_process.terminate_calls == 1


def test_timed_out_job_is_failed_even_if_sigterm_handler_exits_zero(tmp_path: Path) -> None:
    job = _make_local_job(_time_limited_task, tmp_path)
    job._started_at = time.monotonic() - 61  # noqa: SLF001
    fake_process = _FakeProcess(terminate_returncode=0)
    _stage_running_job(LocalExecutor(enforce_time_limits=True), job, fake_process=fake_process)

    assert job.enforce_timeout() == "terminate"
    assert job.state() == "failed"
    assert "Exceeded the requested" in str(job.failure.reason)


def test_timed_out_job_escalates_once_to_sigkill_after_grace(tmp_path: Path) -> None:
    job = _make_local_job(_time_limited_task, tmp_path)
    job._started_at = time.monotonic() - 61  # noqa: SLF001
    fake_process = _FakeProcess(terminate_returncode=None)
    _stage_running_job(LocalExecutor(enforce_time_limits=True), job, fake_process=fake_process)

    assert job.enforce_timeout(kill_grace_s=5) == "terminate"
    assert job.enforce_timeout(kill_grace_s=5) is None
    job._timed_out_at = time.monotonic() - 6  # noqa: SLF001
    assert job.enforce_timeout(kill_grace_s=5) == "kill"
    assert job.enforce_timeout(kill_grace_s=5) is None
    assert fake_process.terminate_calls == 1
    assert fake_process.kill_calls == 1


def test_scheduler_fatal_state_fails_tracked_jobs_and_rejects_submissions(tmp_path: Path) -> None:
    executor = LocalExecutor()
    running = _make_local_job(_time_limited_task, tmp_path)
    queued = _make_local_job(_time_limited_task, tmp_path)
    fake_process = _FakeProcess(terminate_returncode=None)
    _stage_running_job(executor, running, fake_process=fake_process)
    failure = RuntimeError("scheduler bug")

    with executor._scheduler._condition:  # noqa: SLF001
        executor._scheduler._running[running] = None  # noqa: SLF001
        executor._scheduler._ready.append(queued)  # noqa: SLF001
        executor._scheduler._fatal_error = failure  # noqa: SLF001
        executor._scheduler._fail_all_locked("Local scheduler failed: RuntimeError: scheduler bug")  # noqa: SLF001

    assert running.state() == "failed"
    assert queued.state() == "failed"
    assert fake_process.kill_calls == 1
    with pytest.raises(ExecutionError, match="scheduler is unavailable") as raised:
        executor._scheduler.submit(_make_local_job(_time_limited_task, tmp_path))  # noqa: SLF001
    assert raised.value.__cause__ is failure
