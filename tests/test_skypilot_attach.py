"""Durable observation and multi-run session cleanup without a SkyPilot API."""
# ruff: noqa: D103, S101, SLF001

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

import cloudpickle
import msgspec
import pytest

import misen.executors.skypilot as graph_module
from misen import Task, meta
from misen.exceptions import ExecutionError, StatusQueryError, StorageError
from misen.executors.skypilot import GraphWork, LogicalState, RunManifest, RunState, SkyPilotExecutor
from misen.utils.work_unit import WorkUnit
from misen.workspaces.disk import DiskWorkspace

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@meta(id="attach-observe-only", cache=False)
def _task() -> int:
    return 1


@pytest.fixture
def workspace(tmp_path: Path) -> DiskWorkspace:
    return DiskWorkspace(directory=str(tmp_path / "workspace"))


def _stage(workspace: DiskWorkspace) -> None:
    unit = WorkUnit(Task(_task), set())
    node = GraphWork("logical", [], "cpu", ["unused"], {}, "logs/logical.log", unit.resources)
    manifest = RunManifest("durable-run", "snapshot", [node], [])
    workspace.put_job_file("durable-run", "run-manifest.json", msgspec.json.encode(manifest))
    workspace.put_job_file("durable-run", "run-work-units.pkl", cloudpickle.dumps({"logical": unit}))
    _state(workspace, LogicalState(state="running"), heartbeat=time.time())


def _state(workspace: DiskWorkspace, state: LogicalState, *, heartbeat: float, errors: list[str] | None = None) -> None:
    record = RunState("durable-run", {"logical": state}, heartbeat_at=heartbeat, cleanup_errors=errors or [])
    workspace.put_job_file("durable-run", "run-state.json", msgspec.json.encode(record))


def test_attach_observes_and_cancels_without_loading_skypilot(workspace: DiskWorkspace) -> None:
    _stage(workspace)
    jobs = SkyPilotExecutor().attach("durable-run", workspace)
    job = next(iter(jobs.nodes()))
    assert job.coordinator is None
    assert job.job_id == "logical"
    assert job.state() == "running"
    job.cancel()
    cancellation = msgspec.json.decode(workspace.read_job_file("durable-run", "cancellations.json"))
    assert cancellation["job_ids"] == ["logical"]


def test_attach_expired_heartbeat_is_unknown_but_committed_success_stays_done(workspace: DiskWorkspace) -> None:
    _stage(workspace)
    job = next(iter(SkyPilotExecutor().attach("durable-run", workspace).nodes()))
    _state(workspace, LogicalState(state="running"), heartbeat=0)
    assert job.state() == "unknown"
    _state(workspace, LogicalState(state="done"), heartbeat=0)
    assert job.state() == "done"


def test_attach_reports_unresolved_cleanup(workspace: DiskWorkspace) -> None:
    _stage(workspace)
    _state(workspace, LogicalState(state="done"), heartbeat=0, errors=["Native cancellation unresolved."])
    job = next(iter(SkyPilotExecutor().attach("durable-run", workspace).nodes()))
    with pytest.raises(StatusQueryError, match="unresolved cleanup"):
        job.state()


@pytest.mark.parametrize("malformation", ["json", "version", "dependency", "units"])
def test_attach_rejects_malformed_durable_records(workspace: DiskWorkspace, malformation: str) -> None:
    _stage(workspace)
    if malformation == "units":
        workspace.put_job_file("durable-run", "run-work-units.pkl", cloudpickle.dumps({"logical": object()}))
    else:
        record = msgspec.json.decode(workspace.read_job_file("durable-run", "run-manifest.json"))
        if malformation == "version":
            record["version"] = 2
        elif malformation == "dependency":
            record["nodes"][0]["dependencies"] = ["missing-parent"]
        data = b"not json" if malformation == "json" else msgspec.json.encode(record)
        workspace.put_job_file("durable-run", "run-manifest.json", data)
    with pytest.raises(StorageError):
        SkyPilotExecutor().attach("durable-run", workspace)


def _run(close: Callable[[], None]) -> SimpleNamespace:
    return SimpleNamespace(close=close)


def test_session_closes_all_runs_after_one_cleanup_fails() -> None:
    closed = []

    def fail() -> None:
        closed.append("first")
        msg = "First run cleanup failed."
        raise ExecutionError(msg)

    executor = SkyPilotExecutor(manage_api_server=False)
    with pytest.raises(ExecutionError, match="First run"):  # noqa: PT012 -- test context-manager cleanup
        with executor.session():
            owner = graph_module._runs.get()
            assert owner is not None
            owner[1].extend([_run(fail), _run(lambda: closed.append("second"))])
    assert closed == ["first", "second"]
    assert graph_module._runs.get() is None


def test_session_retains_original_error_and_adds_cleanup_notes() -> None:
    def fail() -> None:
        msg = "Cleanup failed."
        raise ExecutionError(msg)

    with pytest.raises(ValueError, match="original") as caught:  # noqa: PT012 -- preserve body error through cleanup
        with SkyPilotExecutor(manage_api_server=False).session():
            owner = graph_module._runs.get()
            assert owner is not None
            owner[1].append(_run(fail))
            msg = "original"
            raise ValueError(msg)
    assert caught.value.__notes__ == ["Cleanup failed."]
