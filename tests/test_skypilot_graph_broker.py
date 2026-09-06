"""Scoped cluster SDK operations and bounded namespace RPC lifetimes."""
# ruff: noqa: D103, S101, SLF001

from __future__ import annotations

import json
import os
import struct
import threading
import time
from enum import Enum
from multiprocessing.connection import Pipe
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

import misen.executors.skypilot as broker
import misen.executors.skypilot as server
from misen.exceptions import ExecutionError
from misen.executors.skypilot import ManagedSkyPilotSession

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from pathlib import Path


@pytest.mark.parametrize("single", [True, False])
def test_cluster_exec_shares_managed_resource_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, single: bool
) -> None:
    session = ManagedSkyPilotSession(tmp_path)
    call = MagicMock(return_value="exec-request")
    monkeypatch.setattr(session, "call", call)
    sky = session.client
    resource = sky.Resources(infra="aws/us-east-1", cpus="2+")
    task = sky.Task(name="agent", resources=resource if single else [resource], run="agent-command")
    assert sky.exec(task, cluster_name="misen-cpu") == "exec-request"
    call.assert_called_once_with(
        "cluster_exec",
        cluster_name="misen-cpu",
        task={"name": "agent", "resources": [{"infra": "aws/us-east-1", "cpus": "2+"}], "run": "agent-command"},
    )
    serialized = call.call_args.kwargs["task"]
    json.dumps(serialized)
    sky.jobs.launch(task, name="agent")
    assert call.call_args.kwargs["task"] == serialized
    assert task.options["resources"] == (resource if single else [resource])


def test_cluster_methods_are_distinct_from_managed_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = ManagedSkyPilotSession(tmp_path)
    call = MagicMock(return_value="request")
    monkeypatch.setattr(session, "call", call)
    assert session.client.job_status("misen-cpu", [12]) == "request"
    call.assert_called_with("cluster_job_status", cluster_name="misen-cpu", job_ids=[12])
    assert session.client.cancel("misen-cpu", [12]) == "request"
    call.assert_called_with("cluster_cancel", cluster_name="misen-cpu", job_ids=[12])
    assert session.client.queue("misen-cpu", skip_finished=False, all_users=False) == "request"
    call.assert_called_with("cluster_queue", cluster_name="misen-cpu", skip_finished=False, all_users=False)
    session.client.jobs.cancel(job_ids=[42])
    call.assert_called_with("cancel", job_ids=[42])


def test_broker_reconstructs_cluster_task_without_mutating_arguments() -> None:
    sky = SimpleNamespace(
        Resources=MagicMock(side_effect=lambda **kwargs: ("resource", kwargs)),
        Task=MagicMock(side_effect=SimpleNamespace),
        exec=MagicMock(return_value="exec-request"),
    )
    arguments = {"task": {"run": "agent", "resources": [{"cpus": "1+"}]}, "cluster_name": "misen-cpu"}
    assert broker._dispatch(sky, "cluster_exec", arguments) == "exec-request"
    sky.Resources.assert_called_once_with(cpus="1+")
    assert sky.exec.call_args.args[0].resources == [("resource", {"cpus": "1+"})]
    assert sky.exec.call_args.kwargs == {"cluster_name": "misen-cpu"}
    assert arguments["task"]["resources"] == [{"cpus": "1+"}]


@pytest.mark.parametrize(("operation", "method"), [("cluster_job_status", "job_status"), ("cluster_cancel", "cancel")])
def test_broker_cluster_operations_forward_only_explicit_job_ids(operation: str, method: str) -> None:
    target = MagicMock(return_value="request")
    sky = SimpleNamespace(**{method: target})
    assert broker._dispatch(sky, operation, {"cluster_name": "misen-cpu", "job_ids": [12]}) == "request"
    target.assert_called_once_with(cluster_name="misen-cpu", job_ids=[12])


@pytest.mark.parametrize("native_id", [12, None, [12, 13]])
def test_get_never_transports_cluster_or_managed_handles(native_id: int | list[int] | None) -> None:
    sky = SimpleNamespace(get=MagicMock(return_value=(native_id, object())))
    result = broker._dispatch(sky, "get", {"request_id": "exec-request"})
    assert server._decode_result(json.loads(json.dumps(broker._encode_result(result)))) == (native_id, None)


def test_cluster_status_enum_values_round_trip_as_json() -> None:
    class Status(Enum):
        RUNNING = "RUNNING"

    sky = SimpleNamespace(get=MagicMock(return_value={12: Status.RUNNING}))
    result = broker._dispatch(sky, "get", {"request_id": "status-request"})
    assert server._decode_result(broker._encode_result(result)) == {"12": "RUNNING"}


@pytest.mark.parametrize("job_ids", [None, [], [True], [0], [-1], "12"])
def test_broker_rejects_unscoped_cluster_cancellation(job_ids: Any) -> None:
    sky = SimpleNamespace(cancel=MagicMock())
    with pytest.raises(ValueError, match="job IDs"):
        broker._dispatch(sky, "cluster_cancel", {"cluster_name": "misen-cpu", "job_ids": job_ids})
    sky.cancel.assert_not_called()


def test_broker_rejects_all_users_and_all_jobs_switches() -> None:
    sky = SimpleNamespace(cancel=MagicMock(), queue=MagicMock())
    with pytest.raises(ValueError, match="explicit job IDs"):
        broker._dispatch(sky, "cluster_cancel", {"cluster_name": "misen-cpu", "job_ids": [12], "all": True})
    with pytest.raises(ValueError, match="scoped"):
        broker._dispatch(sky, "cluster_queue", {"cluster_name": "misen-cpu", "skip_finished": False, "all_users": True})
    sky.cancel.assert_not_called()
    sky.queue.assert_not_called()


def test_queue_preserves_finished_records_for_exact_name_recovery() -> None:
    sky = SimpleNamespace(queue=MagicMock(return_value="queue-request"))
    assert (
        broker._dispatch(
            sky, "cluster_queue", {"cluster_name": "misen-cpu", "skip_finished": False, "all_users": False}
        )
        == "queue-request"
    )
    sky.queue.assert_called_once_with(cluster_name="misen-cpu", skip_finished=False, all_users=False)


@pytest.mark.parametrize("cluster", ["", "*", "misen-*", "cluster\n", " cluster"])
def test_proxy_rejects_nonexplicit_cluster_names_without_starting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cluster: str
) -> None:
    session = ManagedSkyPilotSession(tmp_path)
    call = MagicMock()
    monkeypatch.setattr(session, "call", call)
    with pytest.raises(ValueError, match="explicit SkyPilot cluster"):
        session.client.cancel(cluster, [1])
    call.assert_not_called()


def _connected_session(tmp_path: Path) -> tuple[ManagedSkyPilotSession, Connection, Connection]:
    session = ManagedSkyPilotSession(tmp_path, _call_timeout_s=0.03)
    client, peer = Pipe(duplex=True)
    session._connection = client
    return session, client, peer


def test_rpc_timeout_discards_connection_and_never_restarts_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, client, peer = _connected_session(tmp_path)
    start = MagicMock(side_effect=AssertionError("must not start replacement broker"))
    monkeypatch.setattr(session, "_start", start)
    try:
        with pytest.raises(ExecutionError, match=r"timed out.*discarded"):
            session.call("get", request_id="slow-request")
        assert json.loads(peer.recv_bytes())["op"] == "get"
        assert client.closed
        assert session._connection is None
        with pytest.raises(ExecutionError, match=r"timed out.*discarded"):
            session.call("api_info")
        session.close()
        assert session.closed
        start.assert_not_called()
    finally:
        client.close()
        peer.close()


def test_close_after_drain_timeout_does_not_start_or_use_a_new_connection(tmp_path: Path) -> None:
    session, client, peer = _connected_session(tmp_path)
    first = SimpleNamespace(
        managed_job_id=None, _terminal_state=None, _resolve_managed_job_id=lambda sky: sky.get("slow-launch")
    )
    second = SimpleNamespace(managed_job_id=None, _terminal_state=None, _resolve_managed_job_id=MagicMock())
    session.jobs.extend([first, second])
    try:
        with pytest.raises(ExecutionError, match=r"cleanup failed.*timed out"):
            session.close()
        assert client.closed
        assert session.closed
        second._resolve_managed_job_id.assert_not_called()
        assert json.loads(peer.recv_bytes())["op"] == "get"
    finally:
        client.close()
        peer.close()


def test_rpc_backend_error_keeps_aligned_connection_usable(tmp_path: Path) -> None:
    session, client, peer = _connected_session(tmp_path)
    session._call_timeout_s = 1

    def respond() -> None:
        peer.recv_bytes()
        peer.send_bytes(b'{"error":"SDK rejected request"}')
        peer.recv_bytes()
        peer.send_bytes(b'{"result":"healthy"}')

    thread = threading.Thread(target=respond)
    thread.start()
    try:
        with pytest.raises(ExecutionError, match="SDK rejected"):
            session.call("api_info")
        assert session.call("api_info") == "healthy"
        assert session._connection is client
    finally:
        thread.join(timeout=1)
        client.close()
        peer.close()


@pytest.mark.parametrize("payload", [b"not json", b"[]", b"{}"])
def test_invalid_rpc_reply_invalidates_connection(tmp_path: Path, payload: bytes) -> None:
    session, client, peer = _connected_session(tmp_path)
    peer.send_bytes(payload)
    try:
        with pytest.raises(ExecutionError, match="Invalid reply"):
            session.call("api_info")
        assert client.closed
        assert session._connection is None
    finally:
        client.close()
        peer.close()


def test_interrupt_discards_in_flight_rpc(tmp_path: Path) -> None:
    session = ManagedSkyPilotSession(tmp_path)
    connection = MagicMock()
    connection.send_bytes.side_effect = KeyboardInterrupt
    connection.fileno.side_effect = OSError
    session._connection = connection
    with pytest.raises(KeyboardInterrupt):
        session.call("get", request_id="interrupted")
    connection.close.assert_called_once()
    assert session._connection is None


def test_partial_reply_frame_cannot_block_session_past_timeout(tmp_path: Path) -> None:
    session, client, peer = _connected_session(tmp_path)
    os.write(peer.fileno(), struct.pack("!i", 1024) + b"partial")
    try:
        with pytest.raises(ExecutionError, match="timed out"):
            session.call("api_info")
        assert client.closed
        session.close()
    finally:
        client.close()
        peer.close()


def test_blocked_request_send_is_also_bounded(tmp_path: Path) -> None:
    session, client, peer = _connected_session(tmp_path)
    try:
        with pytest.raises(ExecutionError, match="timed out"):
            session.call("unused", large="x" * (4 * 1024 * 1024))
        assert client.closed
        session.close()
    finally:
        client.close()
        peer.close()


def test_close_interrupts_active_rpc_within_its_own_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session, client, peer = _connected_session(tmp_path)
    session._call_timeout_s = 10
    monkeypatch.setattr(server, "_STOP_TIMEOUT_S", 0.03)
    failures: list[ExecutionError] = []

    def blocked_call() -> None:
        try:
            session.call("get", request_id="blocked")
        except ExecutionError as exc:
            failures.append(exc)

    thread = threading.Thread(target=blocked_call, daemon=True)
    thread.start()
    try:
        assert peer.poll(1)
        peer.recv_bytes()
        before = time.monotonic()
        with pytest.raises(ExecutionError, match="waiting for an active call"):
            session.close()
        assert time.monotonic() - before < 1
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert failures
        assert client.closed
        assert session.closed
    finally:
        client.close()
        peer.close()


def test_drain_rpc_uses_shared_close_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session, client, peer = _connected_session(tmp_path)
    session._call_timeout_s = 10
    monkeypatch.setattr(server, "_STOP_TIMEOUT_S", 0.03)
    session.jobs.append(
        SimpleNamespace(managed_job_id=None, _terminal_state=None, _resolve_managed_job_id=lambda sky: sky.get("slow"))
    )
    try:
        before = time.monotonic()
        with pytest.raises(ExecutionError, match="cleanup failed"):
            session.close()
        assert time.monotonic() - before < 1
        assert client.closed
    finally:
        client.close()
        peer.close()


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan")])
def test_rpc_timeout_must_be_bounded(tmp_path: Path, timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        ManagedSkyPilotSession(tmp_path, _call_timeout_s=timeout)
