"""Tests for LocalExecutor's ``enforce_time_limits`` feature."""
# ruff: noqa: D103, S101

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

import misen.executors.local as local_module
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


def _make_local_job(task: Callable[[], int], tmp_path: Path) -> LocalJob:
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    work_unit = WorkUnit(root=Task(task), dependencies=set())
    return LocalJob(
        work_unit=work_unit,
        dependencies=set(),
        snapshot=None,
        workspace=workspace,
    )


@dataclass(frozen=True)
class _FakeResult:
    code: int | None = 0
    signal: int | None = None
    timed_out: bool = False

    @property
    def is_success(self) -> bool:
        return self.code == 0 and not self.timed_out


class _FakeProcess:
    """Minimal processkit supervision-session stand-in."""

    def __init__(self, result: _FakeResult | None = None) -> None:
        self.result = result
        self.stop_calls: list[float] = []
        self.status = SimpleNamespace(is_active=result is None, pid=99999)

    def wait(self) -> SimpleNamespace:
        return SimpleNamespace(final_result=self.result)

    def stop(self, grace_seconds: float) -> SimpleNamespace:
        self.stop_calls.append(grace_seconds)
        if self.result is None:
            self.result = _FakeResult(code=None, signal=9)
        self.status.is_active = False
        return SimpleNamespace(final_result=self.result)

    async def astop(self, grace_seconds: float) -> SimpleNamespace:
        return self.stop(grace_seconds)


def test_local_executor_default_does_not_enforce_time_limits() -> None:
    executor = LocalExecutor()
    assert executor.enforce_time_limits is False
    assert executor._scheduler.enforce_time_limits is False  # noqa: SLF001


def test_local_executor_propagates_enforce_time_limits_to_scheduler() -> None:
    executor = LocalExecutor(enforce_time_limits=True)
    assert executor.enforce_time_limits is True
    assert executor._scheduler.enforce_time_limits is True  # noqa: SLF001


def _stage_running_job(job: LocalJob, *, fake_process: _FakeProcess) -> None:
    job._process = cast("Any", fake_process)  # noqa: SLF001
    job._cached_state = "running"  # noqa: SLF001


def test_active_managed_process_remains_running(tmp_path: Path) -> None:
    job = _make_local_job(_time_limited_task, tmp_path)
    fake_process = _FakeProcess()
    _stage_running_job(job, fake_process=fake_process)

    assert job.state() == "running"
    assert fake_process.stop_calls == []


def test_timed_out_result_fails_even_with_zero_exit_code(tmp_path: Path) -> None:
    job = _make_local_job(_time_limited_task, tmp_path)
    fake_process = _FakeProcess(_FakeResult(code=0, timed_out=True))
    fake_process.status.is_active = False
    _stage_running_job(job, fake_process=fake_process)

    assert job.state() == "failed"
    assert job.failure.reason == "Exceeded the requested 1 minute time limit."


def test_force_kill_uses_zero_grace_and_records_reason(tmp_path: Path) -> None:
    job = _make_local_job(_time_limited_task, tmp_path)
    fake_process = _FakeProcess()
    _stage_running_job(job, fake_process=fake_process)

    assert job.force_kill(reason="scheduler failed") is True
    assert fake_process.stop_calls == [0]
    assert job.state() == "failed"
    assert job.failure.reason == "scheduler failed"


def test_stop_publishes_the_terminal_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    job = _make_local_job(_time_limited_task, tmp_path)
    job.log_path = tmp_path / "job.log"
    finalized: list[Path] = []
    monkeypatch.setattr(job.workspace, "finalize_job_log", finalized.append)
    _stage_running_job(job, fake_process=_FakeProcess())

    assert job.stop(grace_seconds=0, reason="executor stopped") is True

    assert finalized == [job.log_path]
    assert job.state() == "failed"


@pytest.mark.parametrize("enforce_time_limits", [False, True])
def test_launch_delegates_optional_timeout_to_processkit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    enforce_time_limits: bool,
) -> None:
    calls: list[tuple[str, object]] = []

    class RecordingCommand:
        def __init__(self, program: str, args: list[str]) -> None:
            calls.append(("command", (program, args)))

        def envs(self, values: dict[str, str]) -> RecordingCommand:
            calls.append(("envs", values))
            return self

        def stdout_file(self, path: Path, *, append: bool) -> RecordingCommand:
            calls.append(("stdout_file", (path, append)))
            return self

        def stderr_file(self, path: Path, *, append: bool) -> RecordingCommand:
            calls.append(("stderr_file", (path, append)))
            return self

        def kill_on_parent_death(self) -> RecordingCommand:
            calls.append(("kill_on_parent_death", None))
            return self

        def cpu_affinity(self, cpus: list[int]) -> RecordingCommand:
            calls.append(("cpu_affinity", cpus))
            return self

        def timeout(self, seconds: float) -> RecordingCommand:
            calls.append(("timeout", seconds))
            return self

        def timeout_grace(self, seconds: float) -> RecordingCommand:
            calls.append(("timeout_grace", seconds))
            return self

    session = SimpleNamespace(status=SimpleNamespace(is_active=True, pid=1234))

    class RecordingSupervisor:
        def __init__(self, command: RecordingCommand, *, restart: str) -> None:
            calls.append(("supervisor", (command, restart)))

        def start(self) -> object:
            return session

    log_path = tmp_path / "job.log"
    monkeypatch.setattr(local_module, "Command", RecordingCommand)
    monkeypatch.setattr(local_module, "Supervisor", RecordingSupervisor)
    monkeypatch.setattr(
        local_module,
        "prepare_live_job",
        lambda **_: ("job-id", ["python", "worker.py"], {}, log_path),
    )
    monkeypatch.setattr(local_module, "runtime_job_running", lambda *_args, **_kwargs: None)
    executor = LocalExecutor(enforce_time_limits=enforce_time_limits)
    job = _make_local_job(_time_limited_task, tmp_path)

    executor._scheduler._launch_job(job, cpu_indices=[0], accelerator_indices=[])  # noqa: SLF001

    timeout_calls = [call for call in calls if call[0] in {"timeout", "timeout_grace"}]
    assert timeout_calls == ([("timeout", 60), ("timeout_grace", 5.0)] if enforce_time_limits else [])


def test_scheduler_fatal_state_fails_tracked_jobs_and_rejects_submissions(tmp_path: Path) -> None:
    executor = LocalExecutor()
    running = _make_local_job(_time_limited_task, tmp_path)
    queued = _make_local_job(_time_limited_task, tmp_path)
    fake_process = _FakeProcess()
    _stage_running_job(running, fake_process=fake_process)
    failure = RuntimeError("scheduler bug")

    with executor._scheduler._condition:  # noqa: SLF001
        executor._scheduler._running[running] = None  # noqa: SLF001
        executor._scheduler._ready.append(queued)  # noqa: SLF001
        executor._scheduler._fatal_error = failure  # noqa: SLF001
        executor._scheduler._fail_all_locked("Local scheduler failed: RuntimeError: scheduler bug")  # noqa: SLF001

    assert running.state() == "failed"
    assert queued.state() == "failed"
    assert fake_process.stop_calls == [0]
    with pytest.raises(ExecutionError, match="scheduler is unavailable") as raised:
        executor._scheduler.submit(_make_local_job(_time_limited_task, tmp_path))  # noqa: SLF001
    assert raised.value.__cause__ is failure
