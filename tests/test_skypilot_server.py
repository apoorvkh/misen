"""Owned API sessions and real subprocess cleanup without cloud operations."""

from __future__ import annotations

import contextlib
import io
import json
import os
import select
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import psutil
import pytest

import misen.utils.skypilot_server as server_module
from misen import Task, meta
from misen.exceptions import ConfigError, ExecutionError, SubmissionError
from misen.executor import Executor
from misen.executors.skypilot import SkyPilotExecutor, SkyPilotJob
from misen.utils.skypilot_server import ManagedSkyPilotSession, active_session, managed_session
from misen.utils.work_unit import WorkUnit
from misen.workspaces.memory import InMemoryWorkspace


@meta(id="skypilot_session_task", cache=False)
def _task() -> int:
    return 1


@pytest.fixture
def fake_server(monkeypatch, tmp_path):
    original_start = MagicMock()
    sky = SimpleNamespace(
        server=SimpleNamespace(
            common=SimpleNamespace(
                is_api_server_local=MagicMock(return_value=True),
                get_server_url=MagicMock(return_value="http://127.0.0.1:46580"),
                check_server_healthy_or_start_fn=original_start,
                check_server_healthy=MagicMock(),
            )
        ),
        skypilot_config=SimpleNamespace(get_nested=MagicMock(return_value=False)),
        get=MagicMock(return_value=([42], None)),
    )
    process = MagicMock()
    process.poll.return_value = None
    process.stdin = io.BytesIO()
    process.stdout = io.BytesIO()
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(server_module.subprocess, "Popen", popen)
    monkeypatch.setattr(server_module.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(ManagedSkyPilotSession, "_wait_ready", lambda self, endpoint: None)
    return SimpleNamespace(sky=sky, process=process, popen=popen, original_start=original_start)


def test_session_is_lazy_and_nested_until_outer_exit(monkeypatch):
    stop = MagicMock()
    monkeypatch.setattr(ManagedSkyPilotSession, "_stop", stop)
    executor = SkyPilotExecutor(manage_api_server=True)
    with executor.session():
        outer = active_session()
        assert outer is not None
        with executor.session():
            assert active_session() is outer
        stop.assert_not_called()
        assert outer.process is None
    assert outer.closed
    assert active_session() is None
    stop.assert_called_once()


def test_external_mode_keeps_existing_sdk_behavior():
    with SkyPilotExecutor().session():
        assert active_session() is None


def test_blocking_submission_opens_session_through_polling(monkeypatch):
    def submit(_self, tasks, workspace, *, blocking):
        assert active_session() is not None
        assert blocking is True
        raise SubmissionError("submission failed")

    monkeypatch.setattr(Executor, "submit", submit)
    with pytest.raises(SubmissionError, match="submission failed"):
        SkyPilotExecutor(manage_api_server=True).submit(set(), InMemoryWorkspace(), blocking=True)
    assert active_session() is None


def test_nonblocking_managed_submission_requires_session():
    with pytest.raises(ConfigError, match="with executor.session"):
        SkyPilotExecutor(manage_api_server=True).submit({Task(_task)}, InMemoryWorkspace())


def test_empty_submission_does_not_start_a_server(fake_server):
    graph = SkyPilotExecutor(manage_api_server=True).submit(set(), InMemoryWorkspace(), blocking=True)
    assert not graph.nodes()
    fake_server.popen.assert_not_called()


def test_concurrent_session_rejected_without_closing_owner():
    with managed_session():
        original = active_session()
        with ThreadPoolExecutor(max_workers=1) as threads:

            def open_session():
                with managed_session():
                    pytest.fail("A concurrent process-wide session must be rejected")

            with pytest.raises(ConfigError, match="another thread"):
                threads.submit(open_session).result()
        assert active_session() is original
        assert not original.closed


@pytest.mark.parametrize("failure", [RuntimeError("body failed"), KeyboardInterrupt(), SystemExit(2)])
def test_cleanup_and_sdk_restoration_after_errors(fake_server, failure):
    with pytest.raises(type(failure)):
        with managed_session():
            session = active_session()
            session.ensure_started(fake_server.sky)
            fake_server.sky.server.common.check_server_healthy_or_start_fn()
            fake_server.original_start.assert_not_called()
            fake_server.sky.server.common.check_server_healthy.assert_called_once()
            raise failure
    assert session.closed
    assert fake_server.process.stdin.closed
    fake_server.process.wait.assert_called_once()
    assert fake_server.sky.server.common.check_server_healthy_or_start_fn is fake_server.original_start


@pytest.mark.parametrize("local,consolidated", [(False, False), (True, True)])
def test_remote_and_consolidated_servers_rejected(fake_server, local, consolidated):
    fake_server.sky.server.common.is_api_server_local.return_value = local
    fake_server.sky.skypilot_config.get_nested.return_value = consolidated
    with managed_session(), pytest.raises(ConfigError):
        active_session().ensure_started(fake_server.sky)
    fake_server.popen.assert_not_called()
    assert fake_server.sky.server.common.check_server_healthy_or_start_fn is fake_server.original_start


def test_startup_failure_stops_guardian_and_restores_sdk(fake_server, monkeypatch):
    def fail_ready(self, endpoint):
        raise ExecutionError("startup timed out")

    monkeypatch.setattr(ManagedSkyPilotSession, "_wait_ready", fail_ready)
    with managed_session(), pytest.raises(ExecutionError, match="startup timed out"):
        active_session().ensure_started(fake_server.sky)
    assert fake_server.process.stdin.closed
    assert fake_server.sky.server.common.check_server_healthy_or_start_fn is fake_server.original_start


def test_mid_session_server_failure_cannot_autostart_replacement(fake_server):
    with managed_session():
        session = active_session()
        session.ensure_started(fake_server.sky)
        fake_server.process.poll.return_value = 1
        with pytest.raises(ExecutionError, match="exited unexpectedly"):
            session.ensure_started(fake_server.sky)
    fake_server.popen.assert_called_once()
    fake_server.original_start.assert_not_called()


def _job(tmp_path):
    return SkyPilotJob(
        work_unit=WorkUnit(root=Task(_task), dependencies=set()),
        job_id="misen-job",
        managed_job_id=None,
        request_id="launch-request",
        submission_id="submission",
        deadline_minutes=1,
        log_path=tmp_path / "worker.log",
        workspace=InMemoryWorkspace(),
    )


def test_exit_resolves_and_persists_accepted_launch_before_stopping(fake_server, tmp_path):
    with managed_session():
        session = active_session()
        session.ensure_started(fake_server.sky)
        job = _job(tmp_path)
        job._bind_record(job.workspace, "record")
        SkyPilotExecutor()._record_job(job, job.workspace, "record")
        assert job.managed_job_id is None

        def get_result(request_id):
            assert not fake_server.process.stdin.closed
            assert request_id == "launch-request"
            return ([42], None)

        fake_server.sky.get.side_effect = get_result
    assert job.managed_job_id == 42
    assert json.loads(job.workspace.read_job_file("jobs", "record.json"))["native_id"] == 42
    assert fake_server.process.stdin.closed
    with pytest.raises(ExecutionError, match="session is closed"):
        job.state()
    with pytest.raises(ExecutionError, match="session is closed"):
        job.cancel()


def test_drain_failure_preserves_original_error_and_stops_server(fake_server, tmp_path, monkeypatch):
    with pytest.raises(SubmissionError, match="original") as raised:
        with managed_session():
            active_session().ensure_started(fake_server.sky)
            _job(tmp_path)
            monkeypatch.setattr(SkyPilotJob, "_resolve_managed_job_id", MagicMock(side_effect=ValueError("drain")))
            raise SubmissionError("original")
    assert "drain" in raised.value.__notes__[0]
    assert fake_server.process.stdin.closed


def test_server_lifecycle_does_not_change_durable_job_identity():
    from misen.utils.hashing import stable_hash

    for pool in (None, "misen-dev"):
        unmanaged = SkyPilotExecutor(pool=pool)
        managed = SkyPilotExecutor(pool=pool, manage_api_server=True)
        assert stable_hash(unmanaged._job_key_identity()) == stable_hash(managed._job_key_identity())


def test_guardian_refuses_existing_server_without_touching_it(monkeypatch, tmp_path):
    constants = pytest.importorskip("sky.skylet.constants")
    monkeypatch.setattr(constants, "API_SERVER_CREATION_LOCK_PATH", str(tmp_path / "server.lock"))
    monkeypatch.setattr(server_module.signal, "signal", MagicMock())
    existing = SimpleNamespace(info={"cmdline": [sys.executable, "-m", "sky.server.server"]})
    monkeypatch.setattr(psutil, "process_iter", lambda _attrs: [existing])
    supervise = MagicMock()
    monkeypatch.setattr(server_module, "_supervise_server", supervise)
    with pytest.raises(RuntimeError, match="already running"):
        server_module._guard_server("http://127.0.0.1:46580")
    supervise.assert_not_called()


def test_guardian_refuses_contended_creation_lock(monkeypatch, tmp_path):
    constants = pytest.importorskip("sky.skylet.constants")
    filelock = pytest.importorskip("filelock")
    lock_path = tmp_path / "server.lock"
    monkeypatch.setattr(constants, "API_SERVER_CREATION_LOCK_PATH", str(lock_path))
    monkeypatch.setattr(server_module.signal, "signal", MagicMock())
    supervise = MagicMock()
    monkeypatch.setattr(server_module, "_supervise_server", supervise)
    with filelock.FileLock(lock_path), pytest.raises(filelock.Timeout):
        server_module._guard_server("http://127.0.0.1:46580")
    supervise.assert_not_called()


def _read_json_line(pipe, timeout=10):
    readable, _, _ = select.select([pipe], [], [], timeout)
    assert readable, "No handshake received from child process"
    return json.loads(pipe.readline())


def _assert_stopped(pid):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(0.1)
    pytest.fail(f"Owned process {pid} is still running")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
@pytest.mark.parametrize("kill_client", [False, True])
def test_guardian_reaps_server_tree_on_pipe_close_or_client_death(tmp_path, kill_client):
    # This is a real process hierarchy, not an SDK mock. The child pretends to
    # be the API server, and starts its own worker. No SkyPilot/cloud is used.
    worker_pid_path = tmp_path / "worker.pid"
    server_code = (
        "import subprocess, sys, time; from pathlib import Path; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        f"Path({str(worker_pid_path)!r}).write_text(str(p.pid)); time.sleep(120)"
    )
    guardian_code = (
        "import os; from misen.utils.skypilot_server import _supervise_server; "
        f"_supervise_server({[sys.executable, '-c', server_code]!r}, env=dict(os.environ))"
    )
    client_code = (
        "import subprocess, sys, time, json; "
        f"p = subprocess.Popen({[sys.executable, '-c', guardian_code]!r}, "
        "stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True); "
        "print(json.dumps({'guardian': p.pid, **json.loads(p.stdout.readline())}), flush=True); "
        "sys.stdin.buffer.read(1); p.stdin.close(); p.wait(timeout=20)"
    )
    client = subprocess.Popen(
        [sys.executable, "-c", client_code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        identity = _read_json_line(client.stdout)
        deadline = time.monotonic() + 10
        while not worker_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        worker_pid = int(worker_pid_path.read_text())
        if kill_client:
            client.kill()
        else:
            client.stdin.close()
        client.wait(timeout=20)
        for pid in (identity["pid"], worker_pid, identity["guardian"]):
            _assert_stopped(pid)
        if not kill_client:
            assert client.returncode == 0, client.stderr.read().decode()
    finally:
        with contextlib.suppress(ProcessLookupError):
            client.kill()
        client.wait(timeout=20)
        for stream in (client.stdin, client.stdout, client.stderr):
            stream.close()
