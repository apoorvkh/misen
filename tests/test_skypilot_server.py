"""Isolated clients, namespace leases, and opt-in real SDK process cleanup."""

from __future__ import annotations

import contextlib
import io
import json
import os
import select
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Pipe
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import urlsplit

import psutil
import pytest

import misen.executors.skypilot as broker
import misen.executors.skypilot as executor_module
import misen.executors.skypilot as server
from misen import Task, meta
from misen.exceptions import ConfigError, ExecutionError, SubmissionError
from misen.executor import Executor
from misen.executors.skypilot import (
    ManagedSkyPilotSession,
    SkyPilotExecutor,
    SkyPilotJob,
    active_session,
    managed_session,
)
from misen.utils.work_unit import WorkUnit
from misen.workspaces.memory import InMemoryWorkspace


@meta(id="skypilot_session_task", cache=False)
def _task() -> int:
    return 1


def _job(tmp_path, managed_job_id=None):
    return SkyPilotJob(
        work_unit=WorkUnit(root=Task(_task), dependencies=set()),
        job_id="misen-job",
        managed_job_id=managed_job_id,
        request_id="launch-request",
        submission_id="submission",
        deadline_minutes=1,
        log_path=tmp_path / "worker.log",
        workspace=InMemoryWorkspace(),
    )


def test_lazy_nested_and_independent_sessions(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    start = MagicMock()
    monkeypatch.setattr(ManagedSkyPilotSession, "ensure_started", start)
    with managed_session("first") as first:
        with managed_session("first") as nested:
            assert nested is first
        with managed_session("second") as second:
            assert second is not first
            assert active_session() is second
        assert second.closed
        assert active_session() is first
        with ThreadPoolExecutor(max_workers=1) as threads:

            def other_thread():
                assert active_session() is None
                with managed_session("first") as concurrent:
                    assert concurrent is not first
                    assert concurrent.directory == first.directory

            threads.submit(other_thread).result()
    assert first.closed
    assert active_session() is None
    assert not tmp_path.exists() or not list(tmp_path.iterdir())
    start.assert_not_called()


def test_managed_client_never_imports_sky_in_parent(monkeypatch):
    import_module = MagicMock(side_effect=AssertionError("parent imported SkyPilot"))
    monkeypatch.setattr(executor_module.importlib, "import_module", import_module)
    with managed_session() as session:
        assert executor_module._load_skypilot() is session.client
    import_module.assert_not_called()


@pytest.mark.parametrize("name", ["", "../other", "/tmp/foo", "a/b", "x" * 65, "a b", ".", ".."])
def test_invalid_namespace_rejected(name):
    with pytest.raises(ValueError, match="api_server_namespace"):
        SkyPilotExecutor(api_server_namespace=name)


def test_capacity_profiles_do_not_change_namespace_session_isolation(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    for source in ({"cluster": "misen-cpu"}, {"pool": "misen-dev"}):
        capacity = {"cpu": source}
        external = SkyPilotExecutor(capacity=capacity, manage_api_server=False)
        first = SkyPilotExecutor(capacity=capacity, api_server_namespace="first")
        second = SkyPilotExecutor(capacity=capacity, api_server_namespace="second")
        assert first.capacity == second.capacity == external.capacity
        with first.session() as first_session:
            assert active_session() is first_session
            with second.session() as second_session:
                assert active_session() is second_session
                assert first_session.directory != second_session.directory
            assert active_session() is first_session
        with external.session():
            assert active_session() is None


def test_child_environment_isolated_without_hiding_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "https://ordinary.example")
    monkeypatch.setenv("SKYPILOT_DB_CONNECTION_URI", "test-database")
    monkeypatch.setenv("SKYPILOT_SERVICE_ACCOUNT_TOKEN", "test-token")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-credential")
    original = dict(os.environ)
    env = server._isolated_environment(tmp_path, "abcd1234", tmp_path / "config.yaml")
    assert dict(os.environ) == original
    assert env["HOME"] == original["HOME"]
    assert env["AWS_ACCESS_KEY_ID"] == "test-credential"
    assert env["SKY_RUNTIME_DIR"] == str(tmp_path)
    assert env["SKYPILOT_USER_ID"] == "abcd1234"
    assert "SKYPILOT_API_SERVER_ENDPOINT" not in env
    assert "SKYPILOT_SERVICE_ACCOUNT_TOKEN" not in env
    assert "SKYPILOT_DB_CONNECTION_URI" not in env


def test_blocking_submission_scopes_cleanup(monkeypatch):
    def submit(_self, tasks, workspace, *, blocking):
        assert active_session() is not None
        assert blocking is True
        raise SubmissionError("submission failed")

    monkeypatch.setattr(Executor, "submit", submit)
    with pytest.raises(SubmissionError, match="submission failed"):
        SkyPilotExecutor(manage_api_server=True).submit(set(), InMemoryWorkspace(), blocking=True)
    assert active_session() is None


def test_nonblocking_requires_matching_session():
    executor = SkyPilotExecutor(manage_api_server=True, api_server_namespace="first")
    workspace = SimpleNamespace(
        supports_job_file_reads=lambda: True,
        bootstrap_transport=lambda: "remote-fetch",
        get_temp_dir=lambda: server.Path(".cache/misen/tmp"),
    )
    with pytest.raises(ConfigError, match="with executor.session"):
        executor._validate_submission(work_graph=None, pending_work_units=[], workspace=workspace)
    with managed_session("second"), pytest.raises(ConfigError, match="with executor.session"):
        executor._validate_submission(work_graph=None, pending_work_units=[], workspace=workspace)
    with managed_session("second"), executor.session() as session:
        assert active_session() is session
        assert session.directory == server.namespace_directory("first")
        executor._validate_submission(work_graph=None, pending_work_units=[], workspace=workspace)


def test_empty_submission_never_starts(monkeypatch):
    start = MagicMock()
    monkeypatch.setattr(ManagedSkyPilotSession, "ensure_started", start)
    assert not SkyPilotExecutor(manage_api_server=True).submit(set(), InMemoryWorkspace(), blocking=True).nodes()
    start.assert_not_called()


def test_failed_startup_closes_bootstrap_pipe(monkeypatch, tmp_path):
    process = MagicMock(stdin=io.BytesIO(), stdout=io.BytesIO(b'{"error":"SDK lacks isolation"}\n'))
    monkeypatch.setattr(server.subprocess, "Popen", MagicMock(return_value=process))
    monkeypatch.setattr(server.select, "select", lambda *args: ([process.stdout], [], []))
    session = ManagedSkyPilotSession(tmp_path)
    with pytest.raises(ConfigError, match="SDK lacks isolation"):
        session._start()
    assert process.stdin.closed
    assert process.stdout.closed
    assert session._connection is None


@pytest.mark.parametrize("acquired", [False, True])
def test_bootstrap_eof_only_stops_unclaimed_broker(acquired):
    leases = broker._Leases(MagicMock())
    leases.acquired = acquired
    read, write = os.pipe()
    os.close(write)
    try:
        leases.bootstrap(read)
        assert leases.stop.is_set() is not acquired
    finally:
        os.close(read)


def test_exit_drains_launch_before_release(monkeypatch, tmp_path):
    events = []
    with managed_session() as session:
        session._connection = MagicMock()
        monkeypatch.setattr(session.client, "get", lambda _request: (events.append("get") or [42], None))
        monkeypatch.setattr(session, "_exchange", lambda *args, **kwargs: events.append("release"))
        job = _job(tmp_path)
        job._bind_record(job.workspace, "record")
        SkyPilotExecutor()._record_job(job, job.workspace, "record")
    assert events == ["get", "release"]
    assert json.loads(job.workspace.read_job_file("jobs", "record.json"))["native_id"] == 42
    session._connection.close.assert_called_once()
    with pytest.raises(ExecutionError, match="session is closed"):
        job.state()
    with pytest.raises(ExecutionError, match="session is closed"):
        job.cancel()


@pytest.mark.parametrize("error", [SubmissionError("body"), KeyboardInterrupt(), SystemExit(2)])
def test_drain_failure_preserves_original_exception(monkeypatch, tmp_path, error):
    with pytest.raises(type(error)) as raised:
        with managed_session() as session:
            session._connection = MagicMock()
            exchange = MagicMock()
            monkeypatch.setattr(session, "_exchange", exchange)
            monkeypatch.setattr(SkyPilotJob, "_resolve_managed_job_id", MagicMock(side_effect=ValueError("drain")))
            _job(tmp_path)
            raise error
    assert "drain" in raised.value.__notes__[0]
    exchange.assert_called_once_with({"op": "release"}, timeout=server._STOP_TIMEOUT_S)
    assert session.closed


def test_poll_and_cancel_use_job_namespace_in_other_threads(monkeypatch, tmp_path):
    with managed_session("first") as first, managed_session("second") as second:
        first_job = _job(tmp_path, 42)
        first_job._api_session = first
        second_job = _job(tmp_path, 42)
        for session, state in ((first, "RUNNING"), (second, "PENDING")):
            monkeypatch.setattr(session.client.jobs, "queue_v2", MagicMock(return_value="queue"))
            monkeypatch.setattr(session.client.jobs, "cancel", MagicMock(return_value="cancel"))
            monkeypatch.setattr(
                session.client, "get", MagicMock(return_value=([{"job_id": 42, "status": state}], None))
            )
        with ThreadPoolExecutor(max_workers=1) as threads:
            assert threads.submit(SkyPilotJob.bulk_state, [first_job, second_job]).result() == {
                first_job: "running",
                second_job: "pending",
            }
            threads.submit(first_job.cancel).result()
        first.client.jobs.cancel.assert_called_once_with(job_ids=[42])
        second.client.jobs.cancel.assert_not_called()


def test_json_protocol_discards_launch_handle_and_preserves_tuple():
    sky = SimpleNamespace(get=MagicMock(return_value=([42], object())))
    result = broker._dispatch(sky, "get", {"request_id": "launch"})
    assert server._decode_result(json.loads(json.dumps(broker._encode_result(result)))) == ([42], None)
    with pytest.raises(ValueError, match="Unsupported"):
        broker._dispatch(sky, "api_stop", {})
    with pytest.raises(TypeError, match="Unsupported"):
        broker._encode_result(object())


def test_pool_status_omits_opaque_handles():
    sky = SimpleNamespace(
        jobs=SimpleNamespace(pool_status=MagicMock(return_value="status-request")),
        get=MagicMock(
            return_value=[
                {
                    "name": "pool",
                    "status": "READY",
                    "replica_info": [{"status": "READY", "handle": object(), "resources_str": "1 CPU"}],
                }
            ]
        ),
    )
    result = broker._dispatch(sky, "pool_status", {})
    assert json.loads(json.dumps(broker._encode_result(result)))[0]["replica_info"] == [
        {"status": "READY", "resources_str": "1 CPU"}
    ]


@pytest.mark.parametrize("single", [True, False])
def test_proxy_launch_is_json_native_and_does_not_wait(monkeypatch, tmp_path, single):
    session = ManagedSkyPilotSession(tmp_path)
    call = MagicMock(return_value="launch-id")
    monkeypatch.setattr(session, "call", call)
    sky = session.client
    resource = sky.Resources(infra="aws", cpus="1+")
    task = sky.Task(name="task", run="true", resources=resource if single else [resource])
    assert sky.jobs.launch(task, name="task", pool="pool") == "launch-id"
    call.assert_called_once_with(
        "launch",
        task={
            "name": "task",
            "run": "true",
            "resources": [{"infra": "aws", "cpus": "1+"}],
        },
        name="task",
        pool="pool",
    )
    json.dumps(call.call_args.kwargs)


@pytest.mark.parametrize("infra", ["aws/us-east-1", ["aws/us-east-1", "aws/us-west-2"]])
def test_executor_resource_options_round_trip_through_proxy(monkeypatch, tmp_path, infra):
    from misen.executors.skypilot import SkyPilotCapacity

    session = ManagedSkyPilotSession(tmp_path)
    call = MagicMock(return_value="launch-id")
    monkeypatch.setattr(session, "call", call)
    sky = session.client
    profile = SkyPilotCapacity(infra=infra, cpus=1, memory=8)
    resources = [sky.Resources(**profile.as_sky_options())]
    task = sky.Task(name="task", run="true", resources=resources)
    assert sky.jobs.launch(task, name="task", pool="pool") == "launch-id"
    options = call.call_args.kwargs["task"]["resources"]
    assert [option["infra"] for option in options] == [infra]
    json.dumps(call.call_args.kwargs)


def test_namespace_check_forwards_selected_infrastructure(monkeypatch, tmp_path):
    session = ManagedSkyPilotSession(tmp_path)
    result = {"default": {"AWS": ["compute", "storage"]}}
    call = MagicMock(return_value=result)
    monkeypatch.setattr(session, "call", call)
    assert session.check(["aws"], verbose=True) == result
    call.assert_called_once_with("check", infra_list=["aws"], verbose=True)


def test_broker_check_uses_async_sdk_and_waits(monkeypatch):
    sdk = SimpleNamespace(check=MagicMock(return_value="check-request"))
    monkeypatch.setitem(sys.modules, "sky.client", SimpleNamespace(sdk=sdk))
    result = {"default": {"AWS": ["compute", "storage"]}}
    sky = SimpleNamespace(get=MagicMock(return_value=result), check=object())
    assert broker._dispatch(sky, "check", {"infra_list": ["aws"], "verbose": True}) == result
    sdk.check.assert_called_once_with(infra_list=("aws",), verbose=True)
    sky.get.assert_called_once_with("check-request")


@pytest.mark.parametrize("infra", ["aws", [], [""], [None]])
def test_namespace_check_rejects_invalid_selection_without_starting(monkeypatch, tmp_path, infra):
    session = ManagedSkyPilotSession(tmp_path)
    start = MagicMock()
    monkeypatch.setattr(session, "ensure_started", start)
    with pytest.raises(ValueError, match="infra_list"):
        session.check(infra)
    start.assert_not_called()


def test_status_dispatch_bypasses_native_process_detector(monkeypatch):
    status = MagicMock(return_value=[{"status": "FAILED"}])
    monkeypatch.setattr(broker, "_api_status", status)
    sky = SimpleNamespace(api_status=MagicMock(side_effect=AssertionError("native process detector")))
    assert broker._dispatch(sky, "api_status", {"request_ids": ["request"]}) == [{"status": "FAILED"}]
    status.assert_called_once_with(request_ids=["request"])
    sky.api_status.assert_not_called()


def _lease(leases):
    client, owner = Pipe(duplex=True)
    threading.Thread(target=leases._read, args=(owner,), daemon=True).start()
    client.send_bytes(b'{"op":"acquire"}')
    assert json.loads(client.recv_bytes())["result"] == "acquired"
    return client


def test_shared_leases_keep_server_until_last_release():
    leases = broker._Leases(lambda operation, arguments: arguments)
    first, second = _lease(leases), _lease(leases)
    first.send_bytes(b'{"op":"release"}')
    assert first.poll(5)
    assert json.loads(first.recv_bytes()) == {"result": None}
    assert not leases.stop.is_set()
    second.send_bytes(b'{"op":"release"}')
    assert leases.stop.wait(5)
    assert not second.poll(0.05)
    leases.finish()
    assert json.loads(second.recv_bytes()) == {"result": None}
    first.close()
    second.close()


def test_client_disconnect_detected_during_blocking_sdk_call():
    entered, finish = threading.Event(), threading.Event()

    def dispatch(operation, arguments):
        entered.set()
        finish.wait(10)

    leases = broker._Leases(dispatch)
    client = _lease(leases)
    try:
        client.send_bytes(b'{"op":"get"}')
        assert entered.wait(5)
        client.close()
        assert leases.stop.wait(5)
    finally:
        finish.set()
        client.close()


def _assert_stopped(pid):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(0.1)
    pytest.fail(f"Owned process {pid} is still running")


_live = pytest.mark.skipif(os.environ.get("MISEN_TEST_SKYPILOT_SERVER") != "1", reason="Opt-in local SDK server tests")


@_live
def test_real_async_request_status_reports_success_and_failure(monkeypatch, tmp_path):
    """Exercise HTTP status against the wrapped server, without creating cloud resources."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    with managed_session("requests") as session:
        sky = session.client
        session.ensure_started()
        env = server._isolated_environment(
            session.directory, (session.directory / "identity").read_text().strip(), session.directory / "config.yaml"
        )
        env.update(
            SKYPILOT_API_SERVER_ENDPOINT=session.endpoint,
            SKYPILOT_API_SERVER_LOCAL_PORT=str(urlsplit(session.endpoint).port),
        )
        # Submit a read-only cluster-status request in a child so the parent
        # still never imports SkyPilot. A fresh namespace has no clusters.
        code = (
            "from misen.executors.skypilot import _load_isolated_sdk; "
            "sky = _load_isolated_sdk(); "
            "sky.server.common.check_server_healthy_or_start_fn = "
            "lambda *a, **k: sky.server.common.check_server_healthy(); "
            "from sky.client import sdk; print(sdk.status())"
        )
        success = subprocess.check_output([sys.executable, "-c", code], env=env, text=True, timeout=30).strip()
        # This fresh namespace has no controller: cancellation fails remotely,
        # after returning an accepted request ID. It cannot cancel another user's job.
        failure = sky.jobs.cancel(job_ids=[1])
        deadline = time.monotonic() + 60
        while True:
            statuses = sky.api_status(request_ids=[success, failure])
            states = {item["request_id"]: item["status"] for item in statuses}
            assert set(states) == {success, failure}
            if states == {success: "SUCCEEDED", failure: "FAILED"}:
                break
            assert time.monotonic() < deadline, states
            time.sleep(0.1)
        assert sky.get(success) == []
        with pytest.raises(ExecutionError, match="ClusterNotUpError"):
            sky.get(failure)
        session.client.api_info()
    assert "sky" not in sys.modules


@_live
def test_real_namespaces_share_leases_restart_and_leave_parent_untouched(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "http://ordinary.example:1234")
    original_env = dict(os.environ)
    ordinary_identity = server.Path.home() / ".sky" / "user_hash"
    original_hash = ordinary_identity.read_bytes() if ordinary_identity.exists() else None
    first = ManagedSkyPilotSession(server.namespace_directory("first"))
    same = ManagedSkyPilotSession(server.namespace_directory("first"))
    other = ManagedSkyPilotSession(server.namespace_directory("other"))
    pids = []
    try:
        with ThreadPoolExecutor(max_workers=3) as threads:
            list(threads.map(lambda session: session.client.api_info(), (first, same, other)))
        first.client.Resources(infra="aws/us-east-1", cpus="1+", memory="2+").validate()
        assert first.client.api_status(request_ids=["nonexistent-request"]) == []
        assert first.endpoint == same.endpoint
        assert first.endpoint != other.endpoint
        for session in (first, other):
            descriptor = json.loads((session.directory / "server.json").read_text())
            pids.extend([descriptor["pid"], descriptor["server_pid"]])
            pids.extend(child.pid for child in psutil.Process(descriptor["server_pid"]).children(recursive=True))
        first.close()
        same.client.api_info()
        other.client.api_info()
        same.close()
        other.client.api_info()
        identity = (first.directory / "identity").read_text()
        with managed_session("first") as restarted:
            restarted.client.api_info()
            assert (restarted.directory / "identity").read_text() == identity
    finally:
        for session in (first, same, other):
            session.close()
    for pid in pids:
        _assert_stopped(pid)
    assert dict(os.environ) == original_env
    assert (ordinary_identity.read_bytes() if ordinary_identity.exists() else None) == original_hash
    assert "sky" not in sys.modules


@_live
def test_real_client_sigkill_stops_owned_tree(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    code = (
        "import json, time; from misen.executors.skypilot import managed_session; "
        "ctx = managed_session('crash'); s = ctx.__enter__(); s.client.api_info(); "
        "print((s.directory / 'server.json').read_text(), flush=True); time.sleep(180)"
    )
    client = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        readable, _, _ = select.select([client.stdout], [], [], 120)
        assert readable, "No server startup handshake"
        line = client.stdout.readline()
        assert line, client.stderr.read().decode()
        descriptor = json.loads(line)
        pids = [descriptor["pid"], descriptor["server_pid"]]
        pids.extend(child.pid for child in psutil.Process(descriptor["server_pid"]).children(recursive=True))
        client.kill()
        client.wait(timeout=10)
        for pid in pids:
            _assert_stopped(pid)
    finally:
        with contextlib.suppress(ProcessLookupError):
            client.kill()
        client.wait(timeout=10)
        client.stdout.close()
        client.stderr.close()


@_live
def test_real_client_death_during_startup_stops_owned_tree(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    code = (
        "from misen.executors.skypilot import managed_session; "
        "ctx = managed_session('early-crash'); s = ctx.__enter__(); s.client.api_info()"
    )
    client = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 30
        owned = []
        while time.monotonic() < deadline and client.poll() is None:
            owned = psutil.Process(client.pid).children(recursive=True)
            if any("--server" in process.cmdline() for process in owned):
                break
            time.sleep(0.02)
        assert owned, "No owned processes started"
        client.kill()
        client.wait(timeout=10)
        for process in owned:
            _assert_stopped(process.pid)
    finally:
        with contextlib.suppress(ProcessLookupError):
            client.kill()
        client.wait(timeout=10)
