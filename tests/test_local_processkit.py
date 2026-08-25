"""End-to-end processkit coverage for the local executor."""
# ruff: noqa: D103, S101

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import TYPE_CHECKING

from processkit import Command, Supervisor, process_is_alive

import misen.executors.local as local_module
from misen import Task, meta
from misen.executors.local import LocalExecutor, LocalJob
from misen.utils.work_unit import WorkUnit
from misen.workspaces.disk import DiskWorkspace

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


@meta(id="local_processkit_test", cache=False)
def _local_processkit_task() -> None:
    return None


def _allowed_cpu() -> int:
    if sched_getaffinity := getattr(os, "sched_getaffinity", None):
        return min(sched_getaffinity(0))
    return 0


def _local_job(tmp_path: Path) -> tuple[LocalJob, DiskWorkspace]:
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    work_unit = WorkUnit(root=Task(_local_processkit_task), dependencies=set())
    return LocalJob(work_unit, dependencies=set(), snapshot=None, workspace=workspace), workspace


def test_local_launch_logs_both_streams_and_reaps_descendants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "job.log"
    code = (
        "import os, subprocess, sys; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "print(f'descendant={child.pid}', flush=True); "
        "get_affinity = getattr(os, 'sched_getaffinity', lambda _pid: ()); "
        "print('affinity=' + ','.join(map(str, sorted(get_affinity(0)))), flush=True); "
        "print('threads=' + os.environ['OMP_NUM_THREADS'], flush=True); "
        "print('devices=' + os.environ['CUDA_VISIBLE_DEVICES'], flush=True); "
        "print('stderr-marker', file=sys.stderr, flush=True)"
    )
    monkeypatch.setattr(
        local_module,
        "prepare_live_job",
        lambda **_: (
            "job-id",
            [sys.executable, "-c", code],
            {"OMP_NUM_THREADS": "99", "CUDA_VISIBLE_DEVICES": "inherited"},
            log_path,
        ),
    )
    monkeypatch.setattr(local_module, "runtime_job_running", lambda *_args, **_kwargs: None)

    job, workspace = _local_job(tmp_path)
    executor = LocalExecutor(cpu_indices=[_allowed_cpu()])

    executor._scheduler._launch_job(job, cpu_indices=[_allowed_cpu()], accelerator_indices=[])  # noqa: SLF001
    deadline = time.monotonic() + 10
    state = job.state()
    while state == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
        state = job.state()

    assert state == "done"
    log = log_path.read_text(encoding="utf-8")
    assert "stderr-marker" in log
    if hasattr(os, "sched_getaffinity"):
        assert f"affinity={_allowed_cpu()}" in log
    assert "threads=1" in log
    assert "devices=\n" in log
    descendant_line = next(line for line in log.splitlines() if line.startswith("descendant="))
    descendant_pid = int(descendant_line.partition("=")[2])
    assert not process_is_alive(descendant_pid)

    workspace.close()


def test_local_job_reports_processkit_timeout(tmp_path: Path) -> None:
    command = Command(sys.executable, ["-c", "import time; time.sleep(30)"]).timeout(0.1).timeout_grace(0.05)
    job, workspace = _local_job(tmp_path)
    job.set_process(Supervisor(command, restart="never").start(), cpu_indices=[], accelerator_indices=[])
    deadline = time.monotonic() + 5
    state = job.state()
    while state == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
        state = job.state()

    assert state == "failed"
    assert "time limit" in (job.failure.reason or "")
    workspace.close()


def test_local_job_supports_async_tree_stop(tmp_path: Path) -> None:
    command = Command(sys.executable, ["-c", "import time; time.sleep(30)"])
    job, workspace = _local_job(tmp_path)
    job.set_process(Supervisor(command, restart="never").start(), cpu_indices=[], accelerator_indices=[])

    stopped = asyncio.run(job.astop(grace_seconds=0.05, reason="test stop"))

    assert stopped
    assert job.state() == "failed"
    workspace.close()
