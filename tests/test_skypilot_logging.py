"""SkyPilot logs survive bootstrap failures and remain readable through workspaces."""

# ruff: noqa: ANN001, ANN201, D103, PLR2004, S101, SLF001
from __future__ import annotations

import base64
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import misen.executors.skypilot as sky_module
from misen import Task
from misen.utils.work_unit import WorkUnit
from tests.test_cloud_workspace import _MemoryCloudWorkspace
from tests.test_skypilot_executor import _chain_task
from tests.test_skypilot_workers import Sky, controller, spec


def test_runtime_event_relay_handles_partial_writes_and_does_not_repeat(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(sky_module, "runtime_event", events.append)
    session = sky_module.LocalSkySession(tmp_path)
    sky_module._event(tmp_path, "Provisioning worker")
    path = tmp_path / "events.jsonl"
    with path.open("ab") as output:
        output.write(b'"Environment ')
    session.poll_events()
    session.poll_events()
    assert events == ["Provisioning worker"]
    with path.open("ab") as output:
        output.write(b'ready"\n')
    session.poll_events()
    assert events == ["Provisioning worker", "Environment ready"]


@pytest.mark.parametrize("corrupt", [False, True])
def test_event_read_failures_do_not_interrupt_controller_cleanup(tmp_path, corrupt):
    path = tmp_path / "events.jsonl"
    if corrupt:
        path.write_bytes(b"invalid json\n\xff\n")
    else:
        path.mkdir()  # Simulate an unreadable event file without relying on permissions.
    session = sky_module.LocalSkySession(tmp_path)
    status = SimpleNamespace(is_active=True)
    session.process = SimpleNamespace(status=status, stop=lambda **kwargs: setattr(status, "is_active", False))
    sky_module.LocalSkySession.close_all([session], timeout=0.01)
    assert (tmp_path / "stop").exists()
    assert not session.active


@pytest.mark.parametrize("provision_fails", [False, True])
def test_bootstrap_and_provisioning_failures_publish_complete_job_logs(tmp_path, monkeypatch, provision_fails):
    workspace = _MemoryCloudWorkspace(backend="s3", bucket=tmp_path.name, cache_dir=str(tmp_path / "writer"))
    unit = WorkUnit(Task(_chain_task, value=1), set())
    work = spec("a")
    work.log_path = str(workspace.get_job_log("a", unit))
    work.command = "printf 'bootstrap-before-python\n'; exit 7"
    sky = Sky()
    c = controller(monkeypatch, [work], store=workspace, sky=sky)

    def connect(handle, name, events, directory):
        source = (
            "from __future__ import annotations\n" + inspect.getsource(sky_module._worker_main) + "\n_worker_main()"
        )
        connection = sky_module._Connection(
            [sys.executable, "-u", "-c", source, str(tmp_path / "lease")], events, name, 0, directory
        )
        send = connection.send
        connection.send = lambda message: send(
            message | {"log": str(tmp_path / "agent.log")} if message["kind"] == "run" else message
        )
        return [connection], ["127.0.0.1"]

    monkeypatch.setattr(sky_module, "_connect", connect)
    if provision_fails:
        original_stream = sky.stream_and_get

        def stream(request, output_stream):
            if not sky.downs:
                output_stream.write("provisioning-provider-diagnostic\n")
                raise RuntimeError("provisioning failed")
            return original_stream(request, output_stream)

        sky.stream_and_get = stream
    try:
        if provision_fails:
            with pytest.raises(RuntimeError, match="provisioning failed"):
                c.run()
        else:
            deadline = time.monotonic() + 10
            while not c.tick():
                assert time.monotonic() < deadline
                time.sleep(0.01)
        c.logs.close()
        assert c.state.jobs["a"].state == "failed"
        reader = _MemoryCloudWorkspace(backend="s3", bucket=tmp_path.name, cache_dir=str(tmp_path / "reader"))
        try:
            job_log = next(reader.job_log_iter(unit)).read_text()
            assert "SkyPilot job a" in job_log
            assert "SkyPilot: failed" in job_log
            if provision_fails:
                assert "provisioning-provider-diagnostic" in job_log
                assert "RuntimeError: provisioning failed" in job_log
            else:
                assert "bootstrap-before-python" in job_log
                assert "SkyPilot request request-0" in job_log
                assert "Worker subprocess exited" in job_log
            shared = next(p for p in reader.job_log_iter() if p.name.endswith("_skypilot.log"))
            assert "-down.log" in shared.read_text()
        finally:
            reader.close()
    finally:
        for worker in c.state.workers:
            worker.close()
        c.logs.close()
        workspace.close()


def test_preparation_and_rank_output_share_one_log_without_losing_the_tail(tmp_path, monkeypatch):
    from tests.test_skypilot_workers import offering

    workspace = _MemoryCloudWorkspace(backend="s3", bucket=tmp_path.name, cache_dir=str(tmp_path / "writer"))
    work = spec("a", nodes=2)
    work.log_path = str(tmp_path / "a.log")
    c = controller(monkeypatch, [work], offerings=[offering(nodes=2)], store=workspace)
    worker = sky_module.WorkerState("worker", 0, state="ready")
    worker.connections = [SimpleNamespace(send=lambda message: None) for _ in range(2)]
    c.state.workers.append(worker)
    c.last_ping = time.monotonic()
    c.state.jobs["a"].state = "running"
    c.state.jobs["a"].cluster = worker.name
    try:
        for rank in (0, 1):
            c.events.put(
                (
                    worker.name,
                    rank,
                    {"kind": "log", "job_id": "prepare-a", "data": base64.b64encode(b"prepared\n").decode()},
                )
            )
            c.events.put(
                (
                    worker.name,
                    rank,
                    {"kind": "log", "job_id": "a", "data": base64.b64encode(f"rank-{rank}-tail\n".encode()).decode()},
                )
            )
            c.events.put((worker.name, rank, {"kind": "finished", "job_id": "a", "code": 0, "seconds": 1}))
            c._poll_jobs()
            assert c.state.jobs["a"].state == ("running" if rank == 0 else "done")
        text = Path(work.log_path).read_text()
        assert text.count("prepared\n") == 2
        assert text.count("rank-0-tail") == text.count("rank-1-tail") == 1
        assert text.index("rank-1-tail") < text.index("SkyPilot: done")
    finally:
        c.logs.close()
        workspace.close()


def test_running_logs_stream_live_and_session_retains_later_teardown(tmp_path):
    workspace = _MemoryCloudWorkspace(
        backend="s3", bucket=tmp_path.name, cache_dir=str(tmp_path / "writer"), log_flush_interval_s=0.02
    )
    reader = _MemoryCloudWorkspace(backend="s3", bucket=tmp_path.name, cache_dir=str(tmp_path / "reader"))
    unit = WorkUnit(Task(_chain_task, value=1), set())
    work = spec("a")
    work.log_path = str(workspace.get_job_log("a", unit))
    logs = sky_module._JobLogs(workspace, tmp_path)
    try:
        logs.add([work])
        assert len(workspace._live_log_uploaders) == 1  # Only the session for queued work.
        native = tmp_path / "logs" / "provision.log"
        native.write_text("provisioned\n")
        logs.sync()
        logs.write("a", b"still running\n")
        deadline = time.monotonic() + 5
        while True:
            paths = list(reader.job_log_iter(unit))
            if paths and "still running" in paths[0].read_text():
                break
            assert time.monotonic() < deadline
            time.sleep(0.02)
        logs.finish("a", "done")
        native.write_text("terminated\n")  # A rewritten native log must not lose its new bytes.
        logs.close()
        text = next(reader.job_log_iter(unit)).read_text()
        assert text.count("provisioned") == text.count("still running") == 1
        shared = next(p for p in reader.job_log_iter() if p.name.endswith("_skypilot.log"))
        assert "terminated" in shared.read_text()
    finally:
        logs.close()
        reader.close()
        workspace.close()


def test_logging_plugin_relocates_backend_logs_in_spawned_process(tmp_path):
    pytest.importorskip("sky")
    root = tmp_path / "project"
    root.mkdir()
    native = ".cache/misen/skypilot/native"
    config = root / "plugins.json"
    config.write_text(json.dumps({"plugins": [{"class": "misen.executors.skypilot._LogPlugin"}]}))
    code = """
from pathlib import Path
from sky.server import plugins
from sky.backends.cloud_vm_ray_backend import CloudVmRayBackend
plugins.load_plugins(plugins.ExtensionContext(context=plugins.PluginContext.EXECUTOR))
backend = CloudVmRayBackend()
path = Path(backend.log_dir)
path.mkdir(parents=True)
(path / 'provision.log').write_text('native provisioning log')
print(path)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
        env=os.environ | {"SKYPILOT_SERVER_PLUGINS_CONFIG": str(config), "MISEN_SKYPILOT_LOG_DIR": native},
    )
    paths = list((root / native).rglob("provision.log"))
    assert len(paths) == 1
    assert paths[0].read_text() == "native provisioning log"
    assert str(paths[0].relative_to(root).parent) in result.stdout
