"""SLURM executor behavior that doesn't require an actual cluster."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest

import misen.executors.slurm as slurm_module
from misen import Task, meta
from misen.executors.slurm import SlurmJob
from misen.utils.work_unit import WorkUnit
from misen.workspace import Workspace

if TYPE_CHECKING:
    from collections.abc import Sequence


@meta(id="slurm_test_task", cache=False)
def _slurm_test_task(x: int = 0) -> int:
    return x


def _make_slurm_job(slurm_id: str, x: int) -> SlurmJob:
    work_unit = WorkUnit(root=Task(_slurm_test_task, x=x), dependencies=set())
    workspace = cast("Workspace", MagicMock(spec=Workspace))
    return SlurmJob(
        work_unit=work_unit,
        job_id=f"job-{x}",
        slurm_job_id=slurm_id,
        log_path=Path("/dev/null"),
        workspace=workspace,
    )


class _RunRecorder:
    """Records every ``subprocess.run`` invocation and replays canned stdout."""

    def __init__(self, replies: dict[str, str]) -> None:
        self._replies = replies
        self.calls: list[Sequence[str]] = []

    def __call__(self, cmd: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        self.calls.append(list(cmd))
        # Match by trailing binary name regardless of /usr/bin/... prefix.
        binary = cmd[0].rsplit("/", 1)[-1]
        stdout = self._replies.get(binary, "")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")


def test_slurm_bulk_state_runs_one_squeue_call_for_many_jobs(monkeypatch) -> None:
    jobs = [_make_slurm_job(slurm_id=str(i), x=i) for i in range(5)]
    squeue_stdout = "\n".join(f"{i} RUNNING" for i in range(5)) + "\n"
    recorder = _RunRecorder({"squeue": squeue_stdout})
    # Pretend the SLURM binaries exist on PATH.
    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(slurm_module.subprocess, "run", recorder)

    states = SlurmJob.bulk_state(jobs)

    # Exactly one squeue call (no per-job fallback to sacct since squeue
    # answered every id).
    assert len(recorder.calls) == 1
    assert recorder.calls[0][0].endswith("squeue")
    # All requested ids are passed in a single comma-joined argument.
    joined_ids = ",".join(sorted(str(i) for i in range(5)))
    assert joined_ids in recorder.calls[0]
    # Every job got the expected state.
    assert all(states[job] == "running" for job in jobs)


def test_slurm_bulk_state_falls_back_to_sacct_for_jobs_squeue_doesnt_know(monkeypatch) -> None:
    jobs = [_make_slurm_job(slurm_id=str(i), x=i) for i in range(3)]
    # squeue only knows about job 0 (still queued/running). Jobs 1 and 2 have
    # already left the controller's queue and only sacct can answer for them.
    squeue_stdout = "0 RUNNING\n"
    sacct_stdout = "1 COMPLETED\n2 FAILED\n"
    recorder = _RunRecorder({"squeue": squeue_stdout, "sacct": sacct_stdout})
    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(slurm_module.subprocess, "run", recorder)

    states = SlurmJob.bulk_state(jobs)

    # One squeue call covering all ids, plus one sacct call for the 2 ids
    # squeue didn't return.
    assert len(recorder.calls) == 2
    assert recorder.calls[0][0].endswith("squeue")
    assert recorder.calls[1][0].endswith("sacct")
    # sacct call only includes the still-unknown ids, sorted.
    assert "1,2" in recorder.calls[1]
    assert "0" not in recorder.calls[1][-2]  # the joined-id arg, not the format spec

    assert states[jobs[0]] == "running"
    assert states[jobs[1]] == "done"
    assert states[jobs[2]] == "failed"


def test_slurm_bulk_state_finalizes_logs_for_terminal_jobs(monkeypatch, tmp_path) -> None:
    jobs = [_make_slurm_job(slurm_id=str(i), x=i) for i in range(2)]
    log0 = tmp_path / "j0.log"
    log1 = tmp_path / "j1.log"
    log0.write_text("done")
    log1.write_text("running")
    jobs[0].log_path = log0
    jobs[1].log_path = log1

    recorder = _RunRecorder({"squeue": "0 COMPLETED\n1 RUNNING\n"})
    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(slurm_module.subprocess, "run", recorder)

    SlurmJob.bulk_state(jobs)

    # Terminal job's workspace.finalize_job_log must have been called for the
    # first job (done) and not the second (still running).
    cast("MagicMock", jobs[0].workspace.finalize_job_log).assert_called_once_with(log0)
    cast("MagicMock", jobs[1].workspace.finalize_job_log).assert_not_called()


def test_slurm_bulk_state_returns_unknown_when_slurm_cli_missing(monkeypatch) -> None:
    jobs = [_make_slurm_job(slurm_id="42", x=0)]

    def missing_cmd(name: str) -> str:
        msg = f"{name} not on PATH"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", missing_cmd)

    states = SlurmJob.bulk_state(jobs)
    assert states[jobs[0]] == "unknown"


def test_slurm_bulk_state_handles_empty_input() -> None:
    assert SlurmJob.bulk_state([]) == {}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("RUNNING", "running"),
        ("PENDING", "pending"),
        ("COMPLETED", "done"),
        ("FAILED", "failed"),
        ("CANCELLED+", "failed"),
        ("CANCELLED by 1234", "failed"),
        ("OUT_OF_MEMORY", "failed"),
        ("WEIRD_STATE", "unknown"),
    ],
)
def test_normalize_slurm_state_strips_annotations(raw: str, expected: str) -> None:
    assert slurm_module._normalize_slurm_state(raw) == expected  # noqa: SLF001
