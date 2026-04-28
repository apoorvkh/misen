"""Tests for LocalExecutor's ``enforce_time_limits`` feature."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from misen import Task, meta
from misen.executors.local import LocalExecutor, LocalJob
from misen.utils.snapshot import NullSnapshot
from misen.utils.work_unit import WorkUnit
from misen.workspaces.disk import DiskWorkspace

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@meta(id="time_limited_local_task", cache=False, resources={"time": 1})
def _time_limited_task() -> int:
    return 1


@meta(id="no_time_limit_local_task", cache=False)
def _no_time_limit_task() -> int:
    return 1


def _make_local_job(task: Callable[[], int], tmp_path: Path) -> LocalJob:
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    work_unit = WorkUnit(root=Task(task), dependencies=set())
    return LocalJob(
        work_unit=work_unit,
        dependencies=set(),
        snapshot=NullSnapshot(),
        workspace=workspace,
    )


class _FakeProcess:
    """Minimal subprocess.Popen stand-in for terminate() observability."""

    def __init__(self) -> None:
        self.pid = 99999
        self.terminate_calls = 0
        self._returncode: int | None = None

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._returncode = -15  # SIGTERM


def test_local_executor_default_does_not_enforce_time_limits() -> None:
    executor = LocalExecutor()
    assert executor.enforce_time_limits is False
    assert executor._scheduler.enforce_time_limits is False  # noqa: SLF001


def test_local_executor_propagates_enforce_time_limits_to_scheduler() -> None:
    executor = LocalExecutor(enforce_time_limits=True)
    assert executor.enforce_time_limits is True
    assert executor._scheduler.enforce_time_limits is True  # noqa: SLF001


def test_time_limit_exceeded_returns_false_without_a_time_limit(tmp_path: Path) -> None:
    job = _make_local_job(_no_time_limit_task, tmp_path)
    job._cached_state = "running"  # noqa: SLF001
    job._started_at = time.monotonic() - 1_000_000  # noqa: SLF001
    assert job.time_limit_exceeded() is False


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
        executor._scheduler._running.add(job)  # noqa: SLF001
        executor._scheduler._terminate_timed_out_locked()  # noqa: SLF001
        executor._scheduler._running.discard(job)  # noqa: SLF001

    assert fake_process.terminate_calls == 1


def test_scheduler_does_not_terminate_running_job_within_time_limit(tmp_path: Path) -> None:
    executor = LocalExecutor(enforce_time_limits=True)
    job = _make_local_job(_time_limited_task, tmp_path)
    job._started_at = time.monotonic()  # noqa: SLF001
    fake_process = _FakeProcess()
    _stage_running_job(executor, job, fake_process=fake_process)

    with executor._scheduler._condition:  # noqa: SLF001
        executor._scheduler._running.add(job)  # noqa: SLF001
        executor._scheduler._terminate_timed_out_locked()  # noqa: SLF001
        executor._scheduler._running.discard(job)  # noqa: SLF001

    assert fake_process.terminate_calls == 0


def test_scheduler_does_not_terminate_job_without_time_limit(tmp_path: Path) -> None:
    executor = LocalExecutor(enforce_time_limits=True)
    job = _make_local_job(_no_time_limit_task, tmp_path)
    job._started_at = time.monotonic() - 1_000_000  # noqa: SLF001
    fake_process = _FakeProcess()
    _stage_running_job(executor, job, fake_process=fake_process)

    with executor._scheduler._condition:  # noqa: SLF001
        executor._scheduler._running.add(job)  # noqa: SLF001
        executor._scheduler._terminate_timed_out_locked()  # noqa: SLF001
        executor._scheduler._running.discard(job)  # noqa: SLF001

    assert fake_process.terminate_calls == 0
