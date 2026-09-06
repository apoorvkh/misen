"""Real-process tests for finite-lived reusable workers."""
# ruff: noqa: D103, S101

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from misen.exceptions import StorageError
from misen.executors.skypilot import (
    attempt_file_name,
    run_worker_agent,
    worker_file_name,
)
from misen.utils.resource_env import narrow_accelerator_environment
from misen.workspaces.disk import DiskWorkspace

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_RUN = "test-run"
_WORKER = "test-worker"


def _put(workspace: DiskWorkspace, name: str, **record: Any) -> None:
    workspace.put_job_file(_RUN, name, json.dumps({"version": 1, "run_id": _RUN, **record}).encode())


def _get(workspace: DiskWorkspace, name: str) -> dict[str, Any]:
    return json.loads(workspace.read_job_file(_RUN, name))


def _wait(check: Callable[[], Any], timeout: float = 3) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = check()
            if result:
                return result
        except FileNotFoundError:
            pass
        time.sleep(0.01)
    pytest.fail("Timed out waiting for worker protocol progress")


@contextlib.contextmanager
def _running_agent(
    workspace: DiskWorkspace, **options: float
) -> Iterator[tuple[str, threading.Thread, list[BaseException]]]:
    failures: list[BaseException] = []
    _put(workspace, worker_file_name(_WORKER, "lease"), worker_id=_WORKER, sequence=0, stop=False)

    def run() -> None:
        try:
            run_worker_agent(
                workspace,
                _RUN,
                _WORKER,
                **{
                    "lease_timeout_s": 2,
                    "shutdown_grace_s": 0.1,
                    "poll_interval_s": 0.01,
                    "max_runtime_s": 5,
                    **options,
                },
            )
        except BaseException as exc:  # noqa: BLE001 -- surface worker thread failures in the parent test
            failures.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        state = _wait(lambda: _get(workspace, worker_file_name(_WORKER, "state")))
        yield state["generation"], thread, failures
    finally:
        _put(workspace, worker_file_name(_WORKER, "lease"), worker_id=_WORKER, sequence=2**31, stop=True)
        thread.join(timeout=3)
        assert not thread.is_alive(), "Worker did not stop within its finite grace period"


def _command(
    workspace: DiskWorkspace,
    generation: str,
    attempt: str,
    script: str,
    **overrides: Any,
) -> None:
    _put(
        workspace,
        worker_file_name(_WORKER, "command"),
        **{
            "worker_id": _WORKER,
            "generation": generation,
            "attempt_id": attempt,
            "job_id": f"job-{attempt}",
            "argv": [sys.executable, "-c", script],
            "env": {},
            "log_path": f"logs/{attempt}.log",
            "execution_timeout_s": 1,
            "setup_timeout_s": 1,
            **overrides,
        },
    )


def _child_marker(workspace: DiskWorkspace, attempt: str, kind: str, state: str) -> str:
    # Direct file publication keeps subprocess startup independent of Misen imports.
    directory = workspace.get_temp_dir().parent / "job_files" / _RUN
    marker = directory / f"attempt-{attempt}.{kind}.json"
    content = json.dumps({"version": 1, "run_id": _RUN, "attempt_id": attempt, "state": state})
    return (
        "from pathlib import Path; "
        f"p = Path({str(marker)!r}); q = p.with_suffix('.tmp'); "
        f"q.write_text({content!r}); q.replace(p); "
    )


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DiskWorkspace:
    monkeypatch.chdir(tmp_path)
    return DiskWorkspace(directory=str(tmp_path / "workspace"))


def test_worker_reuses_agent_with_fresh_environment_and_durable_results(workspace: DiskWorkspace) -> None:
    with _running_agent(workspace) as (generation, _, failures):
        for attempt, env in (("first", {"MISEN_AGENT_ENV_PROBE": "secret"}), ("second", {})):
            script = (
                _child_marker(workspace, attempt, "started", "running")
                + "import os; print(os.environ.get('MISEN_AGENT_ENV_PROBE', 'clean')); "
                + "print(os.environ['MISEN_RUN_ID']); print(os.environ['MISEN_ATTEMPT_ID']); "
                + _child_marker(workspace, attempt, "result", "done")
            )
            _command(workspace, generation, attempt, script, env=env)
            outcome = _wait(lambda current=attempt: _get(workspace, attempt_file_name(current)))
            assert outcome["state"] == "done"
            assert outcome["generation"] == generation
            assert outcome["job_id"] == f"job-{attempt}"
        assert Path("logs/first.log").read_text().splitlines() == ["secret", _RUN, "first"]
        assert Path("logs/second.log").read_text().splitlines() == ["clean", _RUN, "second"]
        assert not failures
    assert _get(workspace, worker_file_name(_WORKER, "state"))["state"] == "stopped"


def test_stale_generation_and_duplicate_attempt_never_execute(workspace: DiskWorkspace) -> None:
    with _running_agent(workspace) as (generation, _, failures):
        _command(workspace, "stale", "duplicate", "raise RuntimeError('must not execute')")
        time.sleep(0.05)
        assert not Path("logs/duplicate.log").exists()
        script = "print('once'); " + _child_marker(workspace, "duplicate", "result", "done")
        _command(workspace, generation, "duplicate", script)
        _wait(lambda: _get(workspace, attempt_file_name("duplicate")))
        time.sleep(0.05)  # The completed command remains visible across multiple polling cycles.
        assert Path("logs/duplicate.log").read_text() == "once\n"
        assert not failures


def test_accepted_attempt_is_not_replayed_after_restart(workspace: DiskWorkspace) -> None:
    _put(workspace, attempt_file_name("accepted", "accepted"), attempt_id="accepted", generation="previous")
    with _running_agent(workspace) as (generation, _, failures):
        _command(workspace, generation, "accepted", "raise RuntimeError('must not replay')")
        time.sleep(0.08)
        assert not Path("logs/accepted.log").exists()
        assert not failures


@pytest.mark.parametrize(("started", "reason"), [(False, "setup"), (True, "execution")])
def test_setup_and_execution_timeouts_are_separate(workspace: DiskWorkspace, *, started: bool, reason: str) -> None:
    with _running_agent(workspace) as (generation, _, failures):
        script = _child_marker(workspace, "timeout", "started", "running") if started else ""
        script += "import time; time.sleep(5)"
        _command(workspace, generation, "timeout", script, execution_timeout_s=0.1, setup_timeout_s=0.1)
        outcome = _wait(lambda: _get(workspace, attempt_file_name("timeout")))
        assert outcome["state"] == "failed"
        assert reason in outcome["reason"]
        assert not failures


def test_execution_clock_does_not_include_environment_setup(workspace: DiskWorkspace) -> None:
    with _running_agent(workspace) as (generation, _, failures):
        script = (
            "import time; time.sleep(0.15); "
            + _child_marker(workspace, "separate", "started", "running")
            + "time.sleep(0.03); "
            + _child_marker(workspace, "separate", "result", "done")
        )
        _command(workspace, generation, "separate", script, execution_timeout_s=0.1, setup_timeout_s=1)
        outcome = _wait(lambda: _get(workspace, attempt_file_name("separate")))
        assert outcome["state"] == "done"
        assert not failures


@pytest.mark.parametrize(("result", "exit_code", "expected"), [(False, 0, "unknown"), (True, 2, "failed")])
def test_success_requires_durable_result_and_clean_process_exit(
    workspace: DiskWorkspace, *, result: bool, exit_code: int, expected: str
) -> None:
    with _running_agent(workspace) as (generation, _, failures):
        script = _child_marker(workspace, "completion", "result", "done") if result else ""
        script += f"raise SystemExit({exit_code})"
        _command(workspace, generation, "completion", script)
        outcome = _wait(lambda: _get(workspace, attempt_file_name("completion")))
        assert outcome["state"] == expected
        if result:
            assert _get(workspace, attempt_file_name("completion", "result"))["state"] == "done"
        assert not failures


def test_repeated_lease_cannot_renew_and_expiry_terminates_process(workspace: DiskWorkspace) -> None:
    with _running_agent(workspace, lease_timeout_s=0.25) as (generation, thread, failures):
        script = "import os, time; print(os.getpid(), flush=True); time.sleep(5)"
        _command(workspace, generation, "lease", script)
        _wait(lambda: Path("logs/lease.log").read_text().strip())
        _put(workspace, worker_file_name(_WORKER, "lease"), worker_id=_WORKER, sequence=0, stop=False)
        outcome = _wait(lambda: _get(workspace, attempt_file_name("lease")))
        assert outcome["state"] == "unknown"
        assert "lease expired" in outcome["reason"]
        thread.join(timeout=1)
        assert not thread.is_alive()
        pid = int(Path("logs/lease.log").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert not failures


def test_stop_lease_cancels_only_this_agents_process_group(workspace: DiskWorkspace) -> None:
    with _running_agent(workspace) as (generation, thread, failures):
        script = "import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        script += "print(os.getpid(), flush=True); time.sleep(5)"
        _command(workspace, generation, "cancel", script)
        pid = int(_wait(lambda: Path("logs/cancel.log").read_text().strip()))
        _put(workspace, worker_file_name(_WORKER, "lease"), worker_id=_WORKER, sequence=1, stop=True)
        outcome = _wait(lambda: _get(workspace, attempt_file_name("cancel")))
        assert outcome["state"] == "unknown"
        assert "requested stop" in outcome["reason"]
        thread.join(timeout=1)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert not failures


def test_cancel_one_attempt_keeps_agent_available_for_the_next(workspace: DiskWorkspace) -> None:
    with _running_agent(workspace) as (generation, thread, failures):
        _command(workspace, generation, "cancel-one", "import time; print('running', flush=True); time.sleep(5)")
        _wait(lambda: Path("logs/cancel-one.log").read_text().strip())
        _put(
            workspace,
            worker_file_name(_WORKER, "lease"),
            worker_id=_WORKER,
            sequence=1,
            stop=False,
            cancel_attempt_id="cancel-one",
        )
        outcome = _wait(lambda: _get(workspace, attempt_file_name("cancel-one")))
        assert outcome["state"] == "failed"
        assert "cancelled by coordinator" in outcome["reason"]
        assert thread.is_alive()
        _command(workspace, generation, "after-cancel", _child_marker(workspace, "after-cancel", "result", "done"))
        outcome = _wait(lambda: _get(workspace, attempt_file_name("after-cancel")))
        assert outcome["state"] == "done"
        assert outcome["generation"] == generation
        assert not failures


def test_lease_storage_failure_fails_closed(workspace: DiskWorkspace, monkeypatch: pytest.MonkeyPatch) -> None:
    with _running_agent(workspace) as (generation, thread, failures):
        _command(workspace, generation, "io", "import time; print('running', flush=True); time.sleep(5)")
        _wait(lambda: Path("logs/io.log").read_text().strip())
        original = type(workspace).read_job_file

        def failing_read(self: DiskWorkspace, run_id: str, name: str) -> bytes:
            if name == worker_file_name(_WORKER, "lease"):
                msg = "storage unavailable"
                raise StorageError(msg)
            return original(self, run_id, name)

        monkeypatch.setattr(type(workspace), "read_job_file", failing_read)
        outcome = _wait(lambda: _get(workspace, attempt_file_name("io")))
        assert outcome["state"] == "unknown"
        assert "could not be read" in outcome["reason"]
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert not failures


def test_lease_renewal_then_stop_is_bounded(workspace: DiskWorkspace) -> None:
    with _running_agent(workspace, lease_timeout_s=0.15) as (_, thread, failures):
        for sequence in range(1, 5):
            time.sleep(0.06)
            _put(workspace, worker_file_name(_WORKER, "lease"), worker_id=_WORKER, sequence=sequence, stop=False)
            assert thread.is_alive()
        assert not failures


def test_maximum_lifetime_applies_even_with_a_valid_lease(workspace: DiskWorkspace) -> None:
    with _running_agent(workspace, max_runtime_s=0.15) as (_, thread, failures):
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert _get(workspace, worker_file_name(_WORKER, "state"))["state"] == "stopped"
        assert not failures


def test_process_group_cleanup_stops_forked_descendants(workspace: DiskWorkspace) -> None:
    with _running_agent(workspace) as (generation, thread, failures):
        descendant = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(5)"
        script = (
            "import subprocess, sys, time; "
            f"p = subprocess.Popen([sys.executable, '-c', {descendant!r}]); "
            "print(p.pid, flush=True); time.sleep(5)"
        )
        _command(workspace, generation, "descendant", script)
        descendant_pid = int(_wait(lambda: Path("logs/descendant.log").read_text().strip()))
        _put(workspace, worker_file_name(_WORKER, "lease"), worker_id=_WORKER, sequence=1, stop=True)
        _wait(lambda: _get(workspace, attempt_file_name("descendant")))
        thread.join(timeout=1)

        def descendant_stopped() -> bool:
            try:
                os.kill(descendant_pid, 0)
            except ProcessLookupError:
                return True
            # Linux may retain an orphan zombie until the container's PID 1 reaps it.
            status = Path(f"/proc/{descendant_pid}/stat")
            return status.exists() and status.read_text().split()[2] == "Z"

        _wait(descendant_stopped)
        assert not failures


@pytest.mark.parametrize("invalid", [{"argv": []}, {"env": {"BAD=NAME": "value"}}, {"setup_timeout_s": True}])
def test_malformed_commands_fail_closed(workspace: DiskWorkspace, invalid: dict[str, Any]) -> None:
    with _running_agent(workspace) as (generation, thread, failures):
        _command(workspace, generation, "bad-command", "raise RuntimeError('must not run')", **invalid)
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert len(failures) == 1
        assert isinstance(failures[0], ValueError)
        assert not Path("logs/bad-command.log").exists()


def test_oversized_command_is_rejected_without_parsing(workspace: DiskWorkspace) -> None:
    with _running_agent(workspace) as (_, thread, failures):
        workspace.put_job_file(_RUN, worker_file_name(_WORKER, "command"), b" " * (1024 * 1024 + 1))
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert len(failures) == 1
        assert "size limit" in str(failures[0])


def test_symlinked_log_directory_cannot_escape_working_directory(workspace: DiskWorkspace, tmp_path: Path) -> None:
    Path("outside").symlink_to(tmp_path.parent, target_is_directory=True)
    with _running_agent(workspace) as (generation, thread, failures):
        _command(workspace, generation, "symlink", "print('must not run')", log_path="outside/escape.log")
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert len(failures) == 1
        assert "escapes" in str(failures[0])


@pytest.mark.parametrize("log_path", ["../escape.log", "/escape.log", "nested/../../escape.log", "bad\\path"])
def test_invalid_log_paths_fail_before_launch(workspace: DiskWorkspace, log_path: str) -> None:
    with _running_agent(workspace) as (generation, thread, failures):
        _command(workspace, generation, "invalid", "raise RuntimeError('must not run')", log_path=log_path)
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert len(failures) == 1
        assert isinstance(failures[0], ValueError)
        with pytest.raises(FileNotFoundError):
            _get(workspace, attempt_file_name("invalid", "accepted"))


@pytest.mark.parametrize("token", ["../bad", "a/b", "a\\b", "", ".", "a" * 129])
def test_protocol_tokens_cannot_escape_namespaces(token: str) -> None:
    with pytest.raises(ValueError, match="identifiers"):
        worker_file_name(token, "state")
    with pytest.raises(ValueError, match="identifiers"):
        attempt_file_name(token)


@pytest.mark.parametrize("assigned", ["2,4", "GPU-first-uuid,GPU-second-uuid", "MIG-GPU-first/1/0,MIG-GPU-second/2/0"])
def test_accelerator_mask_preserves_scheduler_ids(assigned: str) -> None:
    base = {"CUDA_VISIBLE_DEVICES": assigned, "PATH": "/bin"}
    env = narrow_accelerator_environment(
        base,
        {"MISEN_ACCELERATOR_COUNT": "1", "MISEN_ACCELERATOR_TYPE": "cuda", "CUDA_VISIBLE_DEVICES": "0,1,2,3"},
    )
    assert env["CUDA_VISIBLE_DEVICES"] == assigned.split(",", maxsplit=1)[0]
    assert base["CUDA_VISIBLE_DEVICES"] == assigned
    assert "MISEN_ACCELERATOR_COUNT" not in env


def test_cpu_work_hides_maskable_devices_and_does_not_leak_into_next_task() -> None:
    base = {
        "CUDA_VISIBLE_DEVICES": "2,4",
        "HIP_VISIBLE_DEVICES": "1,3",
        "ROCR_VISIBLE_DEVICES": "1,3",
        "ZE_AFFINITY_MASK": "0,1",
    }
    cpu = narrow_accelerator_environment(base, {"MISEN_ACCELERATOR_COUNT": "0", "MISEN_ACCELERATOR_TYPE": "cuda"})
    assert all(cpu[name] == "" for name in base)
    gpu = narrow_accelerator_environment(base, {"MISEN_ACCELERATOR_COUNT": "1", "MISEN_ACCELERATOR_TYPE": "cuda"})
    assert gpu["CUDA_VISIBLE_DEVICES"] == "2"
    assert base["CUDA_VISIBLE_DEVICES"] == "2,4"


@pytest.mark.parametrize(
    "base", [{}, {"CUDA_VISIBLE_DEVICES": ""}, {"CUDA_VISIBLE_DEVICES": "2"}, {"CUDA_VISIBLE_DEVICES": "2,2"}]
)
def test_missing_or_insufficient_gpu_reservation_fails_closed(base: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="scheduler device mask"):
        narrow_accelerator_environment(base, {"MISEN_ACCELERATOR_COUNT": "2", "MISEN_ACCELERATOR_TYPE": "cuda"})


def test_gpu_mask_is_applied_to_real_subprocess(workspace: DiskWorkspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,4")
    with _running_agent(workspace) as (generation, _, failures):
        script = "import os; print(os.environ['CUDA_VISIBLE_DEVICES']); "
        script += _child_marker(workspace, "gpu-mask", "result", "done")
        _command(
            workspace,
            generation,
            "gpu-mask",
            script,
            env={"MISEN_ACCELERATOR_COUNT": "1", "MISEN_ACCELERATOR_TYPE": "cuda"},
        )
        outcome = _wait(lambda: _get(workspace, attempt_file_name("gpu-mask")))
        assert outcome["state"] == "done"
        assert Path("logs/gpu-mask.log").read_text() == "2\n"
        assert not failures


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="Linux fd accounting")
def test_repeated_guarded_attempts_do_not_leak_lifetime_fds(workspace: DiskWorkspace) -> None:
    before = len(tuple(Path("/proc/self/fd").iterdir()))
    with _running_agent(workspace) as (generation, _, failures):
        for index in range(5):
            attempt = f"fds-{index}"
            _command(workspace, generation, attempt, _child_marker(workspace, attempt, "result", "done"))
            outcome = _wait(lambda current=attempt: _get(workspace, attempt_file_name(current)))
            assert outcome["state"] == "done"
        assert not failures
    assert len(tuple(Path("/proc/self/fd").iterdir())) <= before
