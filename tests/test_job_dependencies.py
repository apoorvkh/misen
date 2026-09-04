"""Tests for durable remote-executor dependency markers."""
# ruff: noqa: D103, S101

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import cloudpickle
import pytest

import misen.utils.job_dependencies as dependency_module
from misen import Task, meta
from misen.exceptions import ExecutionError, StorageError
from misen.utils.job_dependencies import dependency_state_name, publish_dependency_state, run_with_dependencies
from misen.utils.work_unit import WorkUnit
from misen.workspaces.memory import InMemoryWorkspace

if TYPE_CHECKING:
    from collections.abc import Callable

    from misen.workspace import Workspace


@meta(id="remote_dependency_payload_probe", cache=False)
def _write_probe(path: str) -> None:
    Path(path).write_text("executed", encoding="utf-8")


def test_dependency_gate_waits_then_publishes_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    executions: list[str] = []
    sleeps: list[float] = []

    def release_parent(delay: float) -> None:
        sleeps.append(delay)
        workspace.put_job_file("submission", dependency_state_name("parent"), b"done")

    monkeypatch.setattr(dependency_module.time, "sleep", release_parent)

    run_with_dependencies(
        lambda: executions.append("child"),
        workspace=workspace,
        submission_id="submission",
        job_id="child",
        dependencies=((("submission", "parent"), "parent task"),),
    )

    assert executions == ["child"]
    assert sleeps == [1.0]
    assert workspace.read_job_file("submission", dependency_state_name("child")) == b"done"


def test_dependency_failure_cascades_without_running_user_code(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    workspace.put_job_file("submission", dependency_state_name("parent"), b"failed")
    executions: list[str] = []

    with pytest.raises(ExecutionError, match=r"parent task \(job_id=parent\) failed"):
        run_with_dependencies(
            lambda: executions.append("child"),
            workspace=workspace,
            submission_id="submission",
            job_id="child",
            dependencies=((("submission", "parent"), "parent task"),),
        )

    assert executions == []
    assert workspace.read_job_file("submission", dependency_state_name("child")) == b"failed"


def test_invalid_dependency_marker_fails_instead_of_waiting_forever(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    workspace.put_job_file("submission", dependency_state_name("parent"), b"")
    executions: list[str] = []

    with pytest.raises(ExecutionError, match="published an invalid state marker"):
        run_with_dependencies(
            lambda: executions.append("child"),
            workspace=workspace,
            submission_id="submission",
            job_id="child",
            dependencies=((("submission", "parent"), "parent task"),),
        )

    assert executions == []
    assert workspace.read_job_file("submission", dependency_state_name("child")) == b"failed"


def test_completed_attempt_is_idempotent(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    workspace.put_job_file("submission", dependency_state_name("child"), b"done")
    executions: list[str] = []

    run_with_dependencies(
        lambda: executions.append("child"),
        workspace=workspace,
        submission_id="submission",
        job_id="child",
        dependencies=((("submission", "missing-parent"), "parent task"),),
    )

    assert executions == []


def test_user_failure_publishes_failure_marker(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))

    def fail() -> None:
        msg = "user code failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="user code failed"):
        run_with_dependencies(
            fail,
            workspace=workspace,
            submission_id="submission",
            job_id="child",
            dependencies=(),
        )

    assert workspace.read_job_file("submission", dependency_state_name("child")) == b"failed"


def test_first_terminal_dependency_state_is_immutable(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))

    assert publish_dependency_state(workspace, "submission", "job", b"failed") == b"failed"
    assert publish_dependency_state(workspace, "submission", "job", b"done") == b"failed"
    assert workspace.read_job_file("submission", dependency_state_name("job")) == b"failed"

    assert publish_dependency_state(workspace, "submission", "other", b"done") == b"done"
    assert publish_dependency_state(workspace, "submission", "other", b"failed") == b"done"
    assert workspace.read_job_file("submission", dependency_state_name("other")) == b"done"


def test_work_unit_payload_serializes_dependency_gate(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    parent = WorkUnit(root=Task(_write_probe, path=str(tmp_path / "unused")), dependencies=set())
    child = WorkUnit(root=Task(_write_probe, path=str(tmp_path / "probe")), dependencies={parent})
    workspace.put_job_file("parent-submission", dependency_state_name("parent-job"), b"done")

    payload = cloudpickle.loads(
        child.as_payload(
            workspace,
            "child-job",
            submission_id="submission",
            dependency_jobs={parent: ("parent-submission", "parent-job")},
        )
    )
    execute = cast("Callable[[], None]", payload["fn"])
    execute()

    assert (tmp_path / "probe").read_text(encoding="utf-8") == "executed"
    assert workspace.read_job_file("submission", dependency_state_name("child-job")) == b"done"


def test_coordination_io_retries_transient_storage_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    delegate = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    workspace = MagicMock(spec=delegate, wraps=delegate)
    read_attempts = 0
    done_attempts = 0
    sleeps: list[float] = []

    def flaky_read(submission_id: str, name: str) -> bytes:
        nonlocal read_attempts
        read_attempts += 1
        if read_attempts == 1:
            msg = "temporary read outage"
            raise StorageError(msg)
        return delegate.read_job_file(submission_id, name)

    def flaky_put(submission_id: str, name: str, data: bytes) -> str:
        nonlocal done_attempts
        if data == b"done":
            done_attempts += 1
            if done_attempts == 1:
                msg = "temporary write outage"
                raise StorageError(msg)
        return delegate.put_job_file(submission_id, name, data)

    workspace.read_job_file.side_effect = flaky_read
    workspace.put_job_file.side_effect = flaky_put
    monkeypatch.setattr(dependency_module.time, "sleep", sleeps.append)

    run_with_dependencies(
        lambda: None,
        workspace=cast("Workspace", workspace),
        submission_id="submission",
        job_id="child",
        dependencies=(),
    )

    assert sleeps == [0.25, 0.25]
    assert delegate.read_job_file("submission", dependency_state_name("child")) == b"done"


def test_failed_success_marker_does_not_publish_false_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    delegate = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    workspace = MagicMock(spec=delegate, wraps=delegate)

    def reject_done(submission_id: str, name: str, data: bytes) -> str:
        if data == b"done":
            msg = "done marker unavailable"
            raise StorageError(msg)
        return delegate.put_job_file(submission_id, name, data)

    workspace.put_job_file.side_effect = reject_done
    monkeypatch.setattr(dependency_module.time, "sleep", lambda _delay: None)

    with pytest.raises(StorageError, match="done marker unavailable"):
        run_with_dependencies(
            lambda: None,
            workspace=cast("Workspace", workspace),
            submission_id="submission",
            job_id="child",
            dependencies=(),
        )

    with pytest.raises(FileNotFoundError):
        delegate.read_job_file("submission", dependency_state_name("child"))


def test_failed_attempt_cannot_rerun_and_flip_to_success(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    workspace.put_job_file("submission", dependency_state_name("child"), b"failed")
    executions: list[str] = []

    with pytest.raises(ExecutionError, match="already recorded as failed"):
        run_with_dependencies(
            lambda: executions.append("child"),
            workspace=workspace,
            submission_id="submission",
            job_id="child",
            dependencies=(),
        )

    assert executions == []
    assert workspace.read_job_file("submission", dependency_state_name("child")) == b"failed"


def test_ambiguous_successful_write_preserves_done_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    delegate = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    workspace = MagicMock(spec=delegate, wraps=delegate)

    def publish_then_disconnect(submission_id: str, name: str, data: bytes) -> str:
        ref = delegate.put_job_file(submission_id, name, data)
        if data == b"done":
            msg = "response lost after publish"
            raise StorageError(msg)
        return ref

    workspace.put_job_file.side_effect = publish_then_disconnect
    monkeypatch.setattr(dependency_module.time, "sleep", lambda _delay: None)

    run_with_dependencies(
        lambda: None,
        workspace=cast("Workspace", workspace),
        submission_id="submission",
        job_id="child",
        dependencies=(),
    )

    assert delegate.read_job_file("submission", dependency_state_name("child")) == b"done"
