"""Real persistent worker subprocesses, including failure and ownership boundaries."""

# ruff: noqa: ANN001, ANN201, D103, PLR2004, S101, SLF001
from __future__ import annotations

import inspect
import base64
import json
import os
import queue
import shlex
import sys
import time
from pathlib import Path

import pytest

from misen.executors.skypilot import _Connection, _worker_main


@pytest.fixture
def worker(tmp_path):
    events = queue.Queue()
    source = "from __future__ import annotations\n" + inspect.getsource(_worker_main) + "\n_worker_main()"
    connection = _Connection([sys.executable, "-u", "-c", source, str(tmp_path / "lease")], events, "test", 0, tmp_path)
    assert events.get(timeout=5)[2]["kind"] == "ready"
    yield connection, events, tmp_path
    connection.close()


def receive(events, kind, job_id=None):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        event = events.get(timeout=5)[2]
        if event["kind"] == kind and (job_id is None or event.get("job_id") == job_id):
            return event
    raise AssertionError(f"No {kind} event")


def run(connection, root, job_id, command, cpus=(0,), **env):
    connection.send(
        dict(kind="run", job_id=job_id, command=command, cpus=list(cpus), env=env, log=str(root / f"{job_id}.log"))
    )


def test_multiple_commands_share_agent_but_isolate_subprocesses(worker):
    connection, events, root = worker
    code = "import os,json; print(json.dumps(dict(pid=os.getpid(),cpus=list(os.sched_getaffinity(0)),gpu=os.getenv('CUDA_VISIBLE_DEVICES'))))"
    command = shlex.join([sys.executable, "-c", code])
    run(connection, root, "a", command, CUDA_VISIBLE_DEVICES="1")
    run(connection, root, "b", command, CUDA_VISIBLE_DEVICES="")
    finished = set()
    while len(finished) < 2:
        event = receive(events, "finished")
        assert event["code"] == 0
        finished.add(event["job_id"])
    a, b = [json.loads((root / f"{name}.log").read_text()) for name in ("a", "b")]
    assert a["pid"] != b["pid"]
    assert a["cpus"] == b["cpus"] == [min(os.sched_getaffinity(0))]
    assert (a["gpu"], b["gpu"]) == ("1", "")
    assert connection.process.poll() is None


def test_cancel_kills_one_process_group_and_keeps_agent_alive(worker):
    connection, events, root = worker
    run(connection, root, "slow", "sleep 60 & wait")
    receive(events, "started", "slow")
    connection.send(dict(kind="cancel", job_id="slow"))
    assert receive(events, "finished", "slow")["code"] != 0
    run(connection, root, "next", "true")
    assert receive(events, "finished", "next")["code"] == 0


def test_eof_kills_running_child(worker):
    connection, events, root = worker
    run(connection, root, "slow", "exec sleep 60")
    pid = receive(events, "started", "slow")["pid"]
    connection.stdin.close()
    connection.process.wait(timeout=5)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_eof_preserves_output_larger_than_one_protocol_chunk(worker):
    connection, events, root = worker
    expected = ("unicode λ without a trailing newline " * 8000).encode()
    code = (
        "import sys,time; from pathlib import Path; "
        f"sys.stdout.buffer.write({expected!r}); sys.stdout.buffer.flush(); "
        f"Path({str(root / 'written')!r}).touch(); time.sleep(60)"
    )
    # Keep the command below the OS argument-size limit.
    script = root / "output.py"
    script.write_text(code)
    run(connection, root, "output", shlex.join([sys.executable, str(script)]))
    deadline = time.monotonic() + 5
    while not (root / "written").exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    connection.close()
    chunks = []
    while not events.empty():
        event = events.get_nowait()[2]
        if event["kind"] == "log":
            chunks.append(base64.b64decode(event["data"]))
    assert b"".join(chunks) == expected


def test_duplicate_dispatch_fails_without_replaying(worker):
    connection, events, root = worker
    run(connection, root, "once", f"echo done >> {shlex.quote(str(root / 'side-effect'))}")
    assert receive(events, "finished", "once")["code"] == 0
    run(connection, root, "once", "echo duplicate")
    connection.process.wait(timeout=5)
    assert connection.process.returncode != 0
    assert (root / "side-effect").read_text() == "done\n"


def test_prepare_materializes_without_executing_payload(monkeypatch, tmp_path):
    import misen.utils.materialize_env as module
    from unittest.mock import MagicMock

    build = MagicMock(return_value=object())
    execute = MagicMock(side_effect=AssertionError("Preparation must not execute user code"))
    monkeypatch.setattr(module, "_materialize_envs", build)
    monkeypatch.setattr(module, "_worker_command", execute)
    monkeypatch.setenv("MISEN_PREPARE_ONLY", "1")
    module.main(project_dir=tmp_path, payload=tmp_path / "unused.pkl", job_log_path=tmp_path / "unused.log")
    build.assert_called_once()
    execute.assert_not_called()
