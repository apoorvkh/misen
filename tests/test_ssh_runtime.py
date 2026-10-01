"""Real subprocess checks for the portable remote SSH agent."""

# ruff: noqa: ANN001, D103, S101, S603
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time

import pytest
from processkit import process_is_alive

from misen.utils.ssh_runtime import agent_command

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="SSH workers require Linux")


def _agent(command, *, duration=10, lease=5):
    process = subprocess.Popen(
        shlex.split(agent_command()),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    message = {
        "command": command,
        "env": {},
        "cpus": [0],
        "capacity": 1,
        "timeout": duration,
        "lease": lease,
        "grace": 0.1,
    }
    process.stdin.write(json.dumps(message) + "\n")
    process.stdin.flush()
    return process


def _wait_file(path):
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert path.exists()


def test_agent_runs_in_inherited_affinity_and_preserves_exit_status():
    command = shlex.join([sys.executable, "-c", "import os,sys; print(sorted(os.sched_getaffinity(0))); sys.exit(7)"])
    with _agent(command) as process:
        assert process.wait(timeout=5) == 7
        assert process.stdout.read().strip() == str([min(os.sched_getaffinity(0))])


@pytest.mark.parametrize("stop", ["cancel", "eof", "lease", "timeout", "signal"])
def test_agent_kills_descendants_on_stop(tmp_path, stop):
    pid_file = tmp_path / "child.pid"
    code = (
        "import subprocess,sys,time,pathlib; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); "
        "time.sleep(60)"
    )
    command = shlex.join([sys.executable, "-c", code])
    with _agent(command, duration=0.8 if stop == "timeout" else 10, lease=0.8 if stop == "lease" else 5) as process:
        _wait_file(pid_file)
        child = int(pid_file.read_text())
        try:
            if stop == "cancel":
                process.stdin.write("cancel\n")
                process.stdin.flush()
            elif stop == "eof":
                process.stdin.close()
            elif stop == "signal":
                process.terminate()
            assert process.wait(timeout=5) == (124 if stop == "timeout" else 130)
            assert not process_is_alive(child)
        finally:
            if process.poll() is None:
                process.kill()


def test_agent_cleans_background_children_after_success(tmp_path):
    pid_file = tmp_path / "child.pid"
    code = (
        "import subprocess,sys,pathlib; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))"
    )
    with _agent(shlex.join([sys.executable, "-c", code])) as process:
        assert process.wait(timeout=5) == 0
        assert not process_is_alive(int(pid_file.read_text()))


def test_agent_rejects_overstated_cpu_capacity():
    with subprocess.Popen(
        shlex.split(agent_command()),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as process:
        stdout, _ = process.communicate(json.dumps({"capacity": len(os.sched_getaffinity(0)) + 1}) + "\n", timeout=5)
        assert process.returncode != 0
        assert "capacity exceeds" in stdout
