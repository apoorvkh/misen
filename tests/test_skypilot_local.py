"""Owned local API lifecycle, without provisioning any cloud resources."""
# ruff: noqa: ANN001, ANN201, D103, PLR2004, S101, SLF001

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cloudpickle
import pytest

import misen.executors.skypilot as local
from misen.exceptions import ConfigError, SubmissionError
from misen.workspaces.memory import InMemoryWorkspace


def fake_process(monkeypatch):
    process = MagicMock(status=SimpleNamespace(is_active=True))
    process.stop.side_effect = lambda **kwargs: setattr(process.status, "is_active", False)
    monkeypatch.setattr(local, "Supervisor", MagicMock(return_value=SimpleNamespace(start=lambda: process)))
    return process


def test_cleanup_signals_every_controller_before_forcing_any(monkeypatch, tmp_path):
    sessions = []
    for name in ("a", "b"):
        directory = tmp_path / name
        directory.mkdir()
        session = local.LocalSkySession(directory)
        session.process = MagicMock(status=SimpleNamespace(is_active=True))
        sessions.append(session)

    def stop(**kwargs):
        assert all((s.directory / "stop").exists() for s in sessions)

    for session in sessions:
        session.process.stop.side_effect = stop
    local.LocalSkySession.close_all(sessions, timeout=0)
    for session in sessions:
        session.process.stop.assert_called_once_with(grace_seconds=2)


@pytest.mark.parametrize("failure", ["preflight", "timeout", "interrupt"])
def test_failed_or_interrupted_start_cleans_owned_process(monkeypatch, tmp_path, failure):
    process = fake_process(monkeypatch)
    session = local.LocalSkySession(tmp_path)
    monkeypatch.setattr(session, "close_all", lambda sessions: local.LocalSkySession.close_all(sessions, timeout=0))
    if failure == "preflight":
        (tmp_path / "ready.error").write_text("No worker type satisfies memory")
        expected = ConfigError
    elif failure == "timeout":
        expected = SubmissionError
    else:

        def interrupt(timeout):
            raise KeyboardInterrupt

        monkeypatch.setattr(session, "_wait_ready", interrupt)
        expected = KeyboardInterrupt
    with pytest.raises(expected):
        session.start(timeout=0)
    process.stop.assert_called_once()
    assert session not in local._SESSIONS


def test_session_overrides_only_child_runtime_and_endpoint(monkeypatch, tmp_path):
    fake_process(monkeypatch)
    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "https://existing.example")
    monkeypatch.setenv("SKY_RUNTIME_DIR", str(tmp_path / "existing-runtime"))
    before = os.environ.copy()
    (tmp_path / "ready").touch()
    session = local.LocalSkySession(tmp_path)
    try:
        session.start(timeout=1)
        connection = json.loads((tmp_path / "connection.json").read_text())
        assert connection["SKYPILOT_API_SERVER_ENDPOINT"].startswith("http://127.0.0.1:")
        assert connection["SKY_RUNTIME_DIR"] == str(tmp_path / "runtime")
        assert connection["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)
        assert dict(os.environ) == before
    finally:
        local.LocalSkySession.close_all([session], timeout=0)


def test_installed_nightly_local_api_start_and_stop_without_cloud(monkeypatch, tmp_path):
    pytest.importorskip("sky")
    monkeypatch.setenv("SKYPILOT_DISABLE_USAGE_COLLECTION", "1")
    monkeypatch.delenv("SKYPILOT_DISABLE_LOCAL_API_SERVER", raising=False)
    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "https://existing.example")
    monkeypatch.setenv("SKY_RUNTIME_DIR", str(tmp_path / "existing-runtime"))
    before = os.environ.copy()
    directory = tmp_path / "owned"
    directory.mkdir()
    # An empty graph exercises API startup, checkpointing, and shutdown without
    # catalog requests or cloud launches. Other tests exercise DAG scheduling.
    payload = dict(
        workspace=InMemoryWorkspace(),
        submission_id="probe",
        specs=[],
        config=SimpleNamespace(
            workers=[],
            accelerator_memory={},
            max_workers=1,
            idle_timeout_minutes=1,
            name_prefix="misen",
            lookahead_seconds=90,
            reuse_workers=True,
        ),
    )
    (directory / "controller.pkl").write_bytes(cloudpickle.dumps(payload))
    session = local.LocalSkySession(directory)
    try:
        session.start(timeout=120)
        assert (directory / "ready").exists()
    finally:
        local.LocalSkySession.close_all([session])
    assert not session.active
    assert dict(os.environ) == before
    assert not (tmp_path / "existing-runtime").exists()
    connection = json.loads((directory / "connection.json").read_text())
    with socket.socket() as listener:
        # No detached local server remains listening after cleanup.
        listener.bind(("127.0.0.1", int(connection["SKYPILOT_API_SERVER_LOCAL_PORT"])))
    assert (directory / "runtime" / ".sky").is_dir()


def test_startup_failure_is_retryable_but_live_graph_is_not_overwritten(monkeypatch, tmp_path):
    from misen.executors.skypilot import STATE_FILE, ControllerState
    from tests.test_skypilot_workers import Store, spec
    import msgspec

    store = Store()
    payload = {"workspace": store, "submission_id": "ABC", "specs": [spec("a")]}
    (tmp_path / "controller.pkl").touch()
    monkeypatch.setattr(cloudpickle, "loads", lambda data: payload)

    def fail():
        raise ConfigError("API could not start")

    monkeypatch.setattr(local, "_load_skypilot", fail)
    with pytest.raises(ConfigError, match="could not start"):
        local.main(tmp_path)
    state = msgspec.json.decode(store.read_job_file("ABC", STATE_FILE), type=ControllerState)
    assert state.jobs["a"].state == "failed"
    assert not state.workers
    store.put_job_file("ABC", STATE_FILE, b"existing live checkpoint")
    (tmp_path / "ready").touch()
    with pytest.raises(ConfigError):
        local.main(tmp_path)
    assert store.read_job_file("ABC", STATE_FILE) == b"existing live checkpoint"


@pytest.mark.skipif(os.name != "posix", reason="Unix parent reparenting; Windows uses a Job Object")
def test_parent_loss_stops_real_api_and_helpers(monkeypatch, tmp_path):
    import subprocess
    import sys
    import time

    psutil = pytest.importorskip("psutil")
    pytest.importorskip("sky")
    monkeypatch.setenv("SKYPILOT_DISABLE_USAGE_COLLECTION", "1")
    monkeypatch.delenv("SKYPILOT_DISABLE_LOCAL_API_SERVER", raising=False)
    payload = dict(
        workspace=InMemoryWorkspace(),
        submission_id="probe",
        specs=[],
        config=SimpleNamespace(
            workers=[],
            accelerator_memory={},
            max_workers=1,
            idle_timeout_minutes=1,
            name_prefix="misen",
            lookahead_seconds=90,
            reuse_workers=True,
        ),
    )
    (tmp_path / "controller.pkl").write_bytes(cloudpickle.dumps(payload))
    # Hold an empty graph until owner loss, exercising the production watchdog
    # and API cleanup without any provider/catalog/VM operations.
    controller_code = """
import sys, time
from pathlib import Path
from misen.executors.skypilot import Controller
from misen.executors.skypilot import main
original = Controller.run
def hold(self):
    while not self.stop_requested():
        time.sleep(0.1)
    original(self)
Controller.run = hold
main(Path(sys.argv[1]))
"""
    owner_code = """
import os, sys, json, socket, time
from pathlib import Path
import psutil
from processkit import Command, Supervisor
root = Path(sys.argv[1])
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
env = {"SKY_RUNTIME_DIR": str(root / "runtime"), "SKYPILOT_API_SERVER_LOCAL_PORT": str(port),
       "SKYPILOT_API_SERVER_ENDPOINT": f"http://127.0.0.1:{port}", "MISEN_SKYPILOT_OWNER_PID": str(os.getpid())}
command = Command(sys.executable, args=["-c", sys.argv[2], str(root)]).envs(env).stdout_file(root / "controller.log").stderr_file(root / "controller.log", append=True)
process = Supervisor(command, restart="never").start()
deadline = time.monotonic() + 90
while not (root / "ready").exists():
    if not process.status.is_active or time.monotonic() >= deadline:
        raise RuntimeError((root / "controller.log").read_text())
    time.sleep(0.1)
children = [(p.pid, p.create_time()) for p in psutil.Process().children(recursive=True)]
(root / "children.json").write_text(json.dumps(children))
os._exit(0)
"""
    processes = []
    try:
        subprocess.run([sys.executable, "-c", owner_code, str(tmp_path), controller_code], check=True, timeout=120)
        for pid, created in json.loads((tmp_path / "children.json").read_text()):
            try:
                process = psutil.Process(pid)
                if process.create_time() == created:
                    processes.append(process)
            except psutil.NoSuchProcess:
                pass

        def living():
            return [p for p in processes if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]

        deadline = time.monotonic() + 40
        while living() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not living(), f"Orphaned SkyPilot processes: {living()}"
        assert (tmp_path / "stop").exists()
    finally:
        for process in processes:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                process.kill()


def test_fresh_runtime_checks_declared_clouds_before_provisioning(tmp_path):
    from misen.executors.skypilot import SkyPilotWorker

    sky = MagicMock()
    sky.Resources.side_effect = lambda *, infra: SimpleNamespace(cloud=infra.upper())
    (tmp_path / "logs").mkdir()
    sky.stream_and_get.return_value = {"default": {"AWS": ["compute", "storage"]}}
    workers = [SkyPilotWorker(cpus=2, memory=4, infra="aws/us-east-1")]
    local._check_cloud_access(sky, workers, tmp_path)
    sky.client.sdk.check.assert_called_once_with(infra_list=("aws",), verbose=False)
    sky.stream_and_get.return_value = {"default": {"AWS": ["storage"]}}
    with pytest.raises(ConfigError, match="not enabled for: AWS"):
        local._check_cloud_access(sky, workers, tmp_path)
