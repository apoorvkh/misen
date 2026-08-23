# ruff: noqa: D100, D103, EM101, N818, S101, TRY003

from __future__ import annotations

import os
import shutil
import traceback
from pathlib import Path

import pytest

import misen.tasks as tasks_module
from misen import SCRATCH_DIR, Task, meta
from misen.exceptions import CacheError, ExecutionError, LockUnavailableError, SerializationError, StorageError
from misen.utils.task_utils import execute_task
from misen.workspaces.memory import InMemoryWorkspace


class _TaskFailure(RuntimeError):
    pass


class _CleanupFailure(RuntimeError):
    pass


@meta(id="test_failure_handling_raises", cache=True)
def _raises_task_error() -> None:
    print("before failure")
    raise _TaskFailure("task exploded")


@meta(id="test_failure_handling_succeeds", cache=True)
def _successful_task() -> int:
    return 42


@meta(id="test_failure_handling_array_result", cache=True)
def _array_result() -> object:
    import numpy as np

    return np.array([1, 2, 3])


@meta(id="test_failure_handling_system_exit", cache=True)
def _system_exit_task() -> None:
    raise SystemExit(0)


@meta(id="test_failure_handling_dep", cache=True)
def _dependency_task() -> int:
    return 1


@meta(id="test_failure_handling_downstream", cache=True)
def _downstream_task(value: int) -> int:
    return value


@meta(id="test_failure_handling_large_output", cache=True)
def _large_output_failure() -> None:
    os.write(1, (b"captured-output\n" * 32_768) + b"CAPTURE-DRAINED\n")
    raise _TaskFailure("after large output")


_scratch_paths: list[Path] = []


@meta(id="test_failure_handling_scratch_failure", cache=False, exclude={"scratch_dir"})
def _scratch_failure(scratch_dir: Path) -> None:
    _scratch_paths.append(scratch_dir)
    raise _TaskFailure("scratch task failed")


@meta(id="test_failure_handling_scratch_success", cache=False, exclude={"scratch_dir"})
def _scratch_success(scratch_dir: Path) -> int:
    _scratch_paths.append(scratch_dir)
    return 7


def _execute(task: Task, workspace: InMemoryWorkspace, scratch_dir: Path | None = None) -> object:
    return execute_task(
        task=task,
        workspace=workspace,
        dependency_results={},
        job_id="job-1",
        scratch_dir=scratch_dir,
    )


def test_execute_task_preserves_user_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    task = Task(_raises_task_error)

    with pytest.raises(_TaskFailure) as raised:
        _execute(task, workspace)

    frame_names = [frame.name for frame in traceback.extract_tb(raised.value.__traceback__)]
    assert "_raises_task_error" in frame_names
    assert any("job_id=job-1" in note for note in raised.value.__notes__)
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err
    task_log = workspace.get_task_log(task, job_id="job-1").read_text()
    assert "before failure" in task_log
    assert "_TaskFailure: task exploded" in task_log


def test_task_traceback_is_appended_after_captured_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    task = Task(_large_output_failure)

    with pytest.raises(_TaskFailure):
        _execute(task, workspace)

    capsys.readouterr()
    task_log = workspace.get_task_log(task, job_id="job-1").read_text()
    assert task_log.index("CAPTURE-DRAINED") < task_log.index("Traceback")


def test_task_system_exit_cannot_report_success(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))

    with pytest.raises(ExecutionError, match="interpreter exit with code 0") as raised:
        _execute(Task(_system_exit_task), workspace)

    assert isinstance(raised.value.__cause__, SystemExit)


def test_cleanup_failures_do_not_replace_task_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    calls: list[str] = []

    def fail_scratch(*, task: Task) -> None:
        del task
        calls.append("scratch")
        raise _CleanupFailure("scratch cleanup failed")

    def fail_log(*, task: Task, job_id: str | None = None) -> None:
        del task, job_id
        calls.append("log")
        raise _CleanupFailure("log cleanup failed")

    monkeypatch.setattr(workspace, "finalize_scratch_dir", fail_scratch)
    monkeypatch.setattr(workspace, "finalize_task_log", fail_log)

    with pytest.raises(_TaskFailure) as raised:
        _execute(Task(_raises_task_error), workspace, tmp_path / "scratch")

    assert calls == ["scratch", "log"]
    notes = "\n".join(raised.value.__notes__)
    assert "scratch cleanup failed" in notes
    assert "log cleanup failed" in notes


def test_cleanup_failure_surfaces_after_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    calls: list[str] = []

    def fail_scratch(*, task: Task) -> None:
        del task
        calls.append("scratch")
        raise _CleanupFailure("scratch cleanup failed")

    def finalize_log(*, task: Task, job_id: str | None = None) -> None:
        del task, job_id
        calls.append("log")

    monkeypatch.setattr(workspace, "finalize_scratch_dir", fail_scratch)
    monkeypatch.setattr(workspace, "finalize_task_log", finalize_log)

    with pytest.raises(_CleanupFailure, match="scratch cleanup failed"):
        _execute(Task(_successful_task), workspace, tmp_path / "scratch")

    assert calls == ["scratch", "log"]


def test_is_running_treats_only_unresolved_dependencies_as_not_running(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    dependency = Task(_dependency_task)
    downstream = Task(_downstream_task, value=dependency.T)

    assert downstream.is_running(workspace) is False


def test_is_running_does_not_hide_lock_backend_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))

    def fail_lock(*_args: object, **_kwargs: object) -> None:
        raise _TaskFailure("lock backend failed")

    monkeypatch.setattr(workspace, "lock", fail_lock)

    with pytest.raises(_TaskFailure, match="lock backend failed"):
        Task(_successful_task).is_running(workspace)


def test_result_finalization_failure_is_not_reported_as_task_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))

    def fail_save(**_kwargs: object) -> None:
        raise SerializationError("result write failed")

    monkeypatch.setattr(tasks_module, "save_task_result", fail_save)
    caplog.set_level("INFO", logger="misen.tasks")

    with pytest.raises(SerializationError, match="result write failed") as raised:
        Task(_successful_task).result(workspace=workspace, compute_if_uncached=True)

    assert any("could not be finalized" in note for note in raised.value.__notes__)
    assert not any("Task finished" in record.message for record in caplog.records)


def test_failed_task_preserves_cleanup_failure_as_a_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    _scratch_paths.clear()
    real_rmtree = shutil.rmtree

    def fail_scratch_cleanup(path: object, *args: object, **kwargs: object) -> None:
        if _scratch_paths and Path(path) == _scratch_paths[-1]:
            raise OSError("scratch cleanup unavailable")
        real_rmtree(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tasks_module.shutil, "rmtree", fail_scratch_cleanup)

    with pytest.raises(_TaskFailure, match="scratch task failed") as raised:
        Task(_scratch_failure, SCRATCH_DIR).result(workspace=workspace, compute_if_uncached=True)

    assert any("scratch cleanup unavailable" in note for note in raised.value.__notes__)


def test_successful_task_surfaces_scratch_cleanup_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    _scratch_paths.clear()
    real_rmtree = shutil.rmtree

    def fail_scratch_cleanup(path: object, *args: object, **kwargs: object) -> None:
        if _scratch_paths and Path(path) == _scratch_paths[-1]:
            raise OSError("scratch cleanup unavailable")
        real_rmtree(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tasks_module.shutil, "rmtree", fail_scratch_cleanup)

    with pytest.raises(StorageError, match="scratch cleanup unavailable") as raised:
        Task(_scratch_success, SCRATCH_DIR).result(workspace=workspace, compute_if_uncached=True)

    assert any("committed successfully" in note for note in raised.value.__notes__)


def test_task_does_not_commit_after_losing_runtime_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    task = Task(_array_result)
    real_lock = workspace.lock

    class _LostLock:
        def context(self, *, blocking: bool = True, timeout: int | None = None) -> object:
            del blocking, timeout
            from contextlib import nullcontext

            return nullcontext(self)

        def is_locked(self) -> bool:
            return False

    def lock(namespace: str, key: str) -> object:
        return _LostLock() if namespace == "task" else real_lock(namespace=namespace, key=key)  # type: ignore[arg-type]

    monkeypatch.setattr(workspace, "lock", lock)

    with pytest.raises(LockUnavailableError, match="Lost the runtime lock"):
        task.result(workspace=workspace, compute_if_uncached=True)

    with pytest.raises(CacheError):
        workspace.get_result_hash(task)
    assert not workspace.results.result_store
