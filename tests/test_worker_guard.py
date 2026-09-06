"""Real process-tree cleanup when the reusable worker agent dies abruptly."""
# ruff: noqa: D103, PLR2004, S101, S603

from __future__ import annotations

import contextlib
import json
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from misen.executors.skypilot import worker_file_name
from misen.workspaces.disk import DiskWorkspace

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def _wait(check: Callable[[], Any], timeout_s: float = 5) -> Any:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            value = check()
            if value:
                return value
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.01)
    pytest.fail("Process-tree cleanup did not finish within its deadline")


def _stopped(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    status = Path(f"/proc/{pid}/stat")
    return status.exists() and status.read_text().split()[2] == "Z"


@contextlib.contextmanager
def _guard(script: str, *, timeout_s: float = 3) -> Iterator[tuple[subprocess.Popen[bytes], int]]:
    read_fd, write_fd = os.pipe()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "misen.executors.skypilot",
            "--worker-guard",
            "--parent-fd",
            str(read_fd),
            "--deadline",
            str(time.monotonic() + timeout_s),
            "--grace-s",
            "0.05",
        ],
        pass_fds=(read_fd,),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    os.close(read_fd)
    try:
        configuration = json.dumps({"argv": [sys.executable, "-c", script], "env": dict(os.environ)}).encode()
        framed = struct.pack("!I", len(configuration)) + configuration
        offset = 0
        while offset < len(framed):
            offset += os.write(write_fd, framed[offset:])
        yield process, write_fd
    finally:
        with contextlib.suppress(OSError):
            os.close(write_fd)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
        process.wait(timeout=1)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


@pytest.mark.parametrize("exit_code", [0, 7])
def test_guard_preserves_payload_exit_and_output(exit_code: int) -> None:
    with _guard(f"print('payload output', flush=True); raise SystemExit({exit_code})") as (process, _):
        stdout, stderr = process.communicate(timeout=3)
        assert process.returncode == exit_code
        assert stdout == b"payload output\n"
        assert stderr == b""


def test_guard_preserves_payload_signal_exit() -> None:
    with _guard("import os, signal; os.kill(os.getpid(), signal.SIGTERM)") as (process, _):
        process.communicate(timeout=3)
        assert process.returncode == -signal.SIGTERM


def test_guard_hard_deadline_does_not_depend_on_parent_progress(tmp_path: Path) -> None:
    marker = tmp_path / "payload.pid"
    script = (
        "import os, signal, time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(20)"
    )
    with _guard(script, timeout_s=0.3) as (process, lifetime_fd):
        pid = int(_wait(marker.read_text))
        process.communicate(timeout=3)
        assert process.returncode == 124
        os.fstat(lifetime_fd)  # Parent is still alive and its lifetime fd is open.
        _wait(lambda: _stopped(pid))


@pytest.mark.parametrize("close_pipe", [False, True])
def test_guard_signal_or_parent_eof_kills_payload_group(tmp_path: Path, *, close_pipe: bool) -> None:
    marker = tmp_path / "payload.pid"
    script = (
        "import os, signal, time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(20)"
    )
    with _guard(script) as (process, lifetime_fd):
        pid = int(_wait(marker.read_text))
        if close_pipe:
            os.close(lifetime_fd)
        else:
            process.terminate()
        process.communicate(timeout=3)
        assert process.returncode == -signal.SIGTERM
        _wait(lambda: _stopped(pid))


def test_guard_finishes_descendants_after_successful_payload_exit(tmp_path: Path) -> None:
    marker = tmp_path / "descendant.pid"
    descendant = "import time; time.sleep(20)"
    script = (
        "import subprocess,sys; from pathlib import Path; "
        f"p = subprocess.Popen([sys.executable, '-c', {descendant!r}]); "
        f"Path({str(marker)!r}).write_text(str(p.pid))"
    )
    with _guard(script) as (process, _):
        pid = int(_wait(marker.read_text))
        process.communicate(timeout=3)
        assert process.returncode == 0
        _wait(lambda: _stopped(pid))


def test_parent_death_during_guard_configuration_cannot_launch_payload() -> None:
    read_fd, write_fd = os.pipe()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "misen.executors.skypilot",
            "--worker-guard",
            "--parent-fd",
            str(read_fd),
            "--deadline",
            str(time.monotonic() + 2),
            "--grace-s",
            "0.05",
        ],
        pass_fds=(read_fd,),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    os.close(read_fd)
    os.write(write_fd, struct.pack("!I", 1024) + b"partial")
    os.close(write_fd)
    stdout, stderr = process.communicate(timeout=3)
    assert process.returncode == 125
    assert stdout == b""
    assert b"Worker process guard failed" in stderr


def test_agent_sigkill_stops_guard_payload_and_term_resistant_grandchild(tmp_path: Path) -> None:
    workspace = DiskWorkspace(directory=str(tmp_path / "workspace"))
    run_id, worker_id = "guard-run", "worker"
    workspace.put_job_file(
        run_id,
        worker_file_name(worker_id, "lease"),
        json.dumps(
            {
                "version": 1,
                "run_id": run_id,
                "worker_id": worker_id,
                "sequence": 1,
                "stop": False,
            }
        ).encode(),
    )
    agent_code = (
        "import sys; from misen.workspaces.disk import DiskWorkspace; "
        "from misen.executors.skypilot import run_worker_agent; "
        "run_worker_agent(DiskWorkspace(directory=sys.argv[1]), 'guard-run', 'worker', "
        "lease_timeout_s=30, shutdown_grace_s=.4, poll_interval_s=.01, max_runtime_s=60)"
    )
    agent = subprocess.Popen(
        [sys.executable, "-c", agent_code, workspace.directory],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True)
    identifiers: dict[str, int] = {}
    try:
        state = _wait(lambda: json.loads(workspace.read_job_file(run_id, worker_file_name(worker_id, "state"))))
        grandchild_ready = tmp_path / "grandchild.ready"
        marker = tmp_path / "payloads.json"
        grandchild = (
            "import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"Path({str(grandchild_ready)!r}).write_text('ready'); time.sleep(20)"
        )
        script = (
            "import os,signal,subprocess,sys,time,json; from pathlib import Path; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"child = subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
            f"ready = Path({str(grandchild_ready)!r});\n"
            "while not ready.exists(): time.sleep(.001)\n"
            "pids = dict(payload=os.getpid(), guard=os.getppid(), grandchild=child.pid); "
            f"Path({str(marker)!r}).write_text(json.dumps(pids)); "
            "time.sleep(20)"
        )
        workspace.put_job_file(
            run_id,
            worker_file_name(worker_id, "command"),
            json.dumps(
                {
                    "version": 1,
                    "run_id": run_id,
                    "worker_id": worker_id,
                    "generation": state["generation"],
                    "attempt_id": "attempt",
                    "job_id": "job",
                    "argv": [sys.executable, "-c", script],
                    "env": {},
                    "log_path": "logs/payload.log",
                    "execution_timeout_s": 30,
                    "setup_timeout_s": 30,
                }
            ).encode(),
        )
        identifiers = _wait(lambda: json.loads(marker.read_text()))
        agent.kill()
        agent.wait(timeout=2)
        for pid in identifiers.values():
            _wait(lambda current=pid: _stopped(current), timeout_s=3)
        assert unrelated.poll() is None
    finally:
        if agent.poll() is None:
            agent.kill()
        agent.wait(timeout=2)
        if agent.stderr is not None:
            agent.stderr.close()
        for pid in identifiers.values():
            if not _stopped(pid):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
        unrelated.kill()
        unrelated.wait(timeout=2)
