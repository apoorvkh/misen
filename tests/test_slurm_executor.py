"""SLURM executor behavior that doesn't require an actual cluster."""
# ruff: noqa: ANN001, D103, FBT001, PLR2004, S101

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest

import misen.executors.slurm as slurm_module
from misen import DASK_CLIENT, Task, meta
from misen.exceptions import JobFailedError, StatusQueryError, StorageError, SubmissionError
from misen.executor import raise_for_failed_jobs
from misen.executors.slurm import SlurmExecutor, SlurmJob
from misen.utils.work_unit import WorkUnit
from misen.workspace import Workspace

if TYPE_CHECKING:
    from collections.abc import Sequence


@meta(id="slurm_test_task", cache=False)
def _slurm_test_task(x: int = 0) -> int:
    return x


@meta(id="slurm_gpu_test_task", cache=False, resources={"accelerators": 2})
def _slurm_gpu_test_task() -> None:
    return None


@meta(id="slurm_multinode_test_task", cache=False, resources={"nodes": 3})
def _slurm_multinode_test_task() -> None:
    return None


@meta(
    id="slurm_gpu_constrained_test_task",
    cache=False,
    resources={"accelerators": 2, "accelerator_memory": 80},
)
def _slurm_gpu_constrained_test_task() -> None:
    return None


@meta(id="slurm_tpu_test_task", cache=False, resources={"accelerators": 1, "accelerator_type": "tpu"})
def _slurm_tpu_test_task() -> None:
    return None


@meta(id="slurm_dask_test_task", cache=False)
def _slurm_dask_test_task(client: object) -> None:
    del client


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


def _dispatch_task(
    task: Task[Any],
    executor: SlurmExecutor,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[list[str], MagicMock]:
    """Dispatch one task through mocked sbatch and return its command."""
    workspace = cast("Workspace", MagicMock(spec=Workspace))
    snapshot = MagicMock()
    snapshot.prepare_job.return_value = ("job-local", ["python", "-m", "worker"], {}, tmp_path / "slurm.log")
    commands: list[list[str]] = []
    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", lambda name: f"/usr/bin/{name}")

    def run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="123\n", stderr="")

    monkeypatch.setattr(slurm_module.subprocess, "run", run)
    executor._dispatch(  # noqa: SLF001
        work_unit=WorkUnit(root=task, dependencies=set()),
        dependencies=set(),
        workspace=workspace,
        snapshot=snapshot,
    )
    return commands[0], snapshot


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


def test_slurm_bulk_state_uses_sacct_when_squeue_query_fails(monkeypatch) -> None:
    job = _make_slurm_job(slurm_id="42", x=0)
    calls: list[str] = []
    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", lambda name: f"/usr/bin/{name}")

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        binary = command[0].rsplit("/", 1)[-1]
        calls.append(binary)
        if binary == "squeue":
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="controller unavailable")
        return subprocess.CompletedProcess(command, 0, stdout="42 COMPLETED\n", stderr="")

    monkeypatch.setattr(slurm_module.subprocess, "run", run)

    assert SlurmJob.bulk_state([job])[job] == "done"
    assert calls == ["squeue", "sacct"]


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


def test_failed_slurm_job_preserves_failure_when_log_finalization_fails(monkeypatch, tmp_path) -> None:
    job = _make_slurm_job(slurm_id="42", x=0)
    job.log_path = tmp_path / "failed.log"
    job.log_path.write_text("failure output")
    cast("MagicMock", job.workspace.finalize_job_log).side_effect = StorageError("object store unavailable")
    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(slurm_module.subprocess, "run", _RunRecorder({"squeue": "42 FAILED\n"}))

    states = SlurmJob.bulk_state([job])

    assert states[job] == "failed"
    assert "SLURM reported state 'FAILED'" in cast("str", job.failure.reason)
    assert "finalizing log" in cast("str", job.failure.reason)
    with pytest.raises(JobFailedError):
        raise_for_failed_jobs(states)


def test_successful_slurm_job_surfaces_log_finalization_failure(monkeypatch, tmp_path) -> None:
    job = _make_slurm_job(slurm_id="42", x=0)
    job.log_path = tmp_path / "done.log"
    job.log_path.write_text("output")
    cast("MagicMock", job.workspace.finalize_job_log).side_effect = StorageError("object store unavailable")
    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(slurm_module.subprocess, "run", _RunRecorder({"squeue": "42 COMPLETED\n"}))

    with pytest.raises(StorageError, match="object store unavailable"):
        SlurmJob.bulk_state([job])


def test_slurm_bulk_state_raises_when_slurm_cli_missing(monkeypatch) -> None:
    jobs = [_make_slurm_job(slurm_id="42", x=0)]

    def missing_cmd(name: str) -> str:
        msg = f"{name} not on PATH"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", missing_cmd)

    with pytest.raises(StatusQueryError, match="squeue") as exc_info:
        SlurmJob.bulk_state(jobs)

    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_slurm_bulk_state_preserves_unknown_when_query_may_recover(monkeypatch) -> None:
    jobs = [_make_slurm_job(slurm_id="42", x=0)]
    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        slurm_module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="controller unavailable",
        ),
    )

    assert SlurmJob.bulk_state(jobs) == {jobs[0]: "unknown"}


def test_slurm_query_timeout_is_bounded(monkeypatch) -> None:
    jobs = [_make_slurm_job(slurm_id="42", x=0)]
    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", lambda name: f"/usr/bin/{name}")

    def timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired("squeue", slurm_module._SLURM_QUERY_TIMEOUT_S)  # noqa: SLF001

    monkeypatch.setattr(slurm_module.subprocess, "run", timeout)

    assert SlurmJob.bulk_state(jobs) == {jobs[0]: "unknown"}


def test_slurm_bulk_state_handles_empty_input() -> None:
    assert SlurmJob.bulk_state([]) == {}


def test_slurm_dispatch_translates_sbatch_failure(monkeypatch, tmp_path) -> None:
    work_unit = WorkUnit(root=Task(_slurm_test_task), dependencies=set())
    workspace = cast("Workspace", MagicMock(spec=Workspace))
    snapshot = MagicMock()
    snapshot.prepare_job.return_value = ("job-local", ["python", "-m", "worker"], {}, tmp_path / "slurm.log")
    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", lambda name: f"/usr/bin/{name}")

    def fail_sbatch(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, ["sbatch"], stderr="scheduler rejected job")

    monkeypatch.setattr(slurm_module.subprocess, "run", fail_sbatch)

    with pytest.raises(SubmissionError, match="scheduler rejected job") as exc_info:
        SlurmExecutor()._dispatch(  # noqa: SLF001
            work_unit=work_unit,
            dependencies=set(),
            workspace=workspace,
            snapshot=snapshot,
        )

    assert isinstance(exc_info.value.__cause__, subprocess.CalledProcessError)


def test_slurm_dispatch_timeout_is_bounded(monkeypatch, tmp_path) -> None:
    work_unit = WorkUnit(root=Task(_slurm_test_task), dependencies=set())
    workspace = cast("Workspace", MagicMock(spec=Workspace))
    snapshot = MagicMock()
    snapshot.prepare_job.return_value = ("job-local", ["python", "-m", "worker"], {}, tmp_path / "slurm.log")
    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", lambda name: f"/usr/bin/{name}")

    def timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired("sbatch", slurm_module._SLURM_SUBMIT_TIMEOUT_S)  # noqa: SLF001

    monkeypatch.setattr(slurm_module.subprocess, "run", timeout)

    with pytest.raises(SubmissionError, match="timed out") as exc_info:
        SlurmExecutor()._dispatch(  # noqa: SLF001
            work_unit=work_unit,
            dependencies=set(),
            workspace=workspace,
            snapshot=snapshot,
        )

    assert isinstance(exc_info.value.__cause__, subprocess.TimeoutExpired)


def test_slurm_dispatch_translates_missing_sbatch(monkeypatch) -> None:
    work_unit = WorkUnit(root=Task(_slurm_test_task), dependencies=set())

    def missing_sbatch(_name: str) -> str:
        msg = "sbatch not on PATH"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", missing_sbatch)
    with pytest.raises(SubmissionError, match="sbatch not on PATH") as exc_info:
        SlurmExecutor()._dispatch(  # noqa: SLF001
            work_unit=work_unit,
            dependencies=set(),
            workspace=cast("Workspace", MagicMock(spec=Workspace)),
            snapshot=None,
        )

    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_slurm_dispatch_delegates_resource_isolation_to_slurm(monkeypatch, tmp_path) -> None:
    work_unit = WorkUnit(root=Task(_slurm_test_task, x=0), dependencies=set())
    workspace = cast("Workspace", MagicMock(spec=Workspace))
    snapshot = MagicMock()
    log_path = tmp_path / "slurm.log"
    snapshot.prepare_job.return_value = ("job-local", ["python", "-m", "worker"], {}, log_path)

    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        slurm_module.subprocess,
        "run",
        lambda *_, **__: subprocess.CompletedProcess(args=[], returncode=0, stdout="123\n", stderr=""),
    )

    SlurmExecutor()._dispatch(  # noqa: SLF001
        work_unit=work_unit,
        dependencies=set(),
        workspace=workspace,
        snapshot=snapshot,
    )

    # SLURM handles accelerator visibility and CPU affinity; no explicit
    # resource assignment enters snapshot or worker arguments.
    assert set(snapshot.prepare_job.call_args.kwargs) == {"work_unit", "workspace"}


def test_slurm_dispatch_kills_jobs_with_invalid_dependencies(monkeypatch, tmp_path) -> None:
    work_unit = WorkUnit(root=Task(_slurm_test_task, x=0), dependencies=set())
    dependency = _make_slurm_job(slurm_id="42", x=1)
    workspace = cast("Workspace", MagicMock(spec=Workspace))
    snapshot = MagicMock()
    snapshot.prepare_job.return_value = ("job-local", ["python", "-m", "worker"], {}, tmp_path / "slurm.log")
    commands: list[list[str]] = []

    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", lambda name: f"/usr/bin/{name}")

    def run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="123\n", stderr="")

    monkeypatch.setattr(slurm_module.subprocess, "run", run)

    executor = SlurmExecutor()
    executor._dispatch(work_unit=work_unit, dependencies=set(), workspace=workspace, snapshot=snapshot)  # noqa: SLF001
    executor._dispatch(  # noqa: SLF001
        work_unit=work_unit,
        dependencies={dependency},
        workspace=workspace,
        snapshot=snapshot,
    )

    assert "--kill-on-invalid-dep=yes" not in commands[0]
    assert "--kill-on-invalid-dep=yes" in commands[1]
    dependency_index = commands[1].index("--dependency")
    assert commands[1][dependency_index + 1] == "afterok:42"


def test_slurm_dispatch_maps_plain_gpu_count_directly(monkeypatch, tmp_path) -> None:
    command, _ = _dispatch_task(Task(_slurm_gpu_test_task), SlurmExecutor(), monkeypatch, tmp_path)

    assert "--gpus-per-node=2" in command


def test_slurm_dispatch_requests_task_nodes(monkeypatch, tmp_path) -> None:
    command, _ = _dispatch_task(Task(_slurm_multinode_test_task), SlurmExecutor(), monkeypatch, tmp_path)

    nodes_index = command.index("--nodes")
    assert command[nodes_index + 1] == "3"
    wrapped = shlex.split(command[command.index("--wrap") + 1])
    assert wrapped[0] == "env"
    assert "OMP_DYNAMIC=FALSE" in wrapped
    assert "MKL_DYNAMIC=FALSE" in wrapped
    assert "OPENBLAS_DYNAMIC=0" in wrapped
    assert not any(value.startswith(("OMP_NUM_THREADS=", "CUDA_VISIBLE_DEVICES=")) for value in wrapped)
    assert wrapped[-3:] == ["python", "-m", "worker"]


def test_slurm_dask_dispatch_bootstraps_one_private_worker_per_node(monkeypatch, tmp_path) -> None:
    task = Task(_slurm_dask_test_task, DASK_CLIENT).with_resources(nodes=2, cpus=3)
    command, _ = _dispatch_task(task, SlurmExecutor(dask_startup_timeout=45), monkeypatch, tmp_path)

    wrapped = shlex.split(command[command.index("--wrap") + 1])
    assert wrapped[:2] == ["bash", "-c"]
    script = wrapped[2]
    assert "MISEN_DASK_ROLE=scheduler" in script
    assert "MISEN_DASK_ROLE=worker" in script
    assert "MISEN_DASK_ROLE=coordinator" not in script
    assert "--nodes=2" in script
    assert "--ntasks=2" in script
    assert "--cpus-per-task=3" in script
    assert "--overlap" in script
    assert "MISEN_DASK_STARTUP_TIMEOUT=45" in script
    assert "MISEN_DASK_MEMORY_GIB=8" in script
    assert "SLURM_PROCID" not in script
    assert not any(line.lstrip().startswith("env ") for line in script.splitlines())


@pytest.mark.parametrize(
    "flag",
    [
        "cpus-per-gpu",
        "cpus-per-tres",
        "gpus",
        "gpus-per-task",
        "gpus-per-socket",
        "gres",
        "mem-per-tres",
        "nodes",
        "ntasks-per-gpu",
        "tres-per-job",
        "tres-per-node",
        "tres-per-socket",
        "tres-per-task",
    ],
)
def test_slurm_rejects_flags_controlled_by_the_executor(flag, monkeypatch, tmp_path) -> None:
    with pytest.raises(ValueError, match="controlled by SlurmExecutor"):
        SlurmExecutor(default_flags={flag: "conflict"})


@pytest.mark.parametrize("timeout", [True, 0, -1, 1.5])
def test_slurm_validates_dask_startup_timeout_eagerly(timeout: Any) -> None:
    with pytest.raises(ValueError, match="dask_startup_timeout must be a positive integer"):
        SlurmExecutor(dask_startup_timeout=timeout)


def test_slurm_dask_debug_log_redacts_the_actual_wrapper(monkeypatch, tmp_path, caplog) -> None:
    task = Task(_slurm_dask_test_task, DASK_CLIENT).with_resources(nodes=2)

    with caplog.at_level(logging.DEBUG, logger=slurm_module.__name__):
        _dispatch_task(task, SlurmExecutor(), monkeypatch, tmp_path)

    message = next(record.message for record in caplog.records if record.message.startswith("sbatch command"))
    assert "bash -c" in message
    assert "<redacted Dask cluster script>" in message
    assert "MISEN_DASK_ROLE" not in message


def test_slurm_rules_resolve_constrained_accelerator(monkeypatch, tmp_path) -> None:
    executor = SlurmExecutor(
        rules=[
            {
                "when": {"accelerator_memory": 80, "accelerator_type": "cuda"},
                "set": {"gpu-type": "a100-80gb"},
            }
        ]
    )

    command, _ = _dispatch_task(Task(_slurm_gpu_constrained_test_task), executor, monkeypatch, tmp_path)

    assert "--gpus-per-node=a100-80gb:2" in command


def test_slurm_rejects_unmapped_gpu_constraints(tmp_path) -> None:
    work_unit = WorkUnit(root=Task(_slurm_gpu_constrained_test_task), dependencies=set())
    workspace = cast("Workspace", MagicMock(spec=Workspace))
    snapshot = MagicMock()
    snapshot.prepare_job.return_value = ("job-local", ["python", "-m", "worker"], {}, tmp_path / "slurm.log")

    with pytest.raises(SubmissionError, match="rules do not cover"):
        SlurmExecutor()._dispatch(  # noqa: SLF001
            work_unit=work_unit,
            dependencies=set(),
            workspace=workspace,
            snapshot=snapshot,
        )


def test_slurm_rejects_tpus(monkeypatch, tmp_path) -> None:
    with pytest.raises(SubmissionError, match="does not support 'tpu'"):
        _dispatch_task(Task(_slurm_tpu_test_task), SlurmExecutor(), monkeypatch, tmp_path)


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


@pytest.mark.parametrize(
    ("raw", "in_queue", "expected"),
    [
        # Still tracked by the controller (seen via squeue): a preempted job on a
        # PreemptMode=REQUEUE cluster is mid-requeue, not a terminal failure.
        ("PREEMPTED", True, "pending"),
        ("PREEMPTED+", True, "pending"),
        ("PREEMPTED by 1234", True, "pending"),
        # Left the queue (answered only by sacct): the preemption was terminal.
        ("PREEMPTED", False, "failed"),
        ("PREEMPTED+", False, "failed"),
    ],
)
def test_normalize_slurm_state_preempted_depends_on_queue_membership(raw: str, in_queue: bool, expected: str) -> None:
    assert slurm_module._normalize_slurm_state(raw, in_queue=in_queue) == expected  # noqa: SLF001


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("REQUEUE_HOLD", "pending"),
        ("REQUEUE_FED", "pending"),
        ("RESV_DEL_HOLD", "pending"),
        ("EXPEDITING", "pending"),
        ("POWER_UP_NODE", "pending"),
        ("REVOKED", "pending"),
        ("SPECIAL_EXIT", "pending"),
        ("SIGNALING", "running"),
        ("STOPPED", "running"),
        ("RESIZING", "running"),
        ("UPDATE_DB", "running"),
        ("LAUNCH_FAILED", "failed"),
        ("RECONFIG_FAIL", "failed"),
    ],
)
def test_normalize_slurm_queue_flags_are_nonterminal(raw: str, expected: str) -> None:
    assert slurm_module._normalize_slurm_state(raw, in_queue=True) == expected  # noqa: SLF001


def test_slurm_bulk_state_preempted_in_queue_is_pending_without_sacct(monkeypatch) -> None:
    """A PREEMPTED job still in squeue is pending and needs no sacct lookup.

    On a PreemptMode=REQUEUE cluster a preempted job transiently reports
    PREEMPTED in squeue before being requeued under the same id. Because the
    controller still tracks it, its state must be resolved from squeue alone --
    consulting sacct (which would report the terminal PREEMPTED) and finalizing
    the log would wrongly abandon a job SLURM is going to rerun.
    """
    job = _make_slurm_job(slurm_id="7", x=0)
    # sacct *would* report the terminal PREEMPTED, but it must never be reached.
    recorder = _RunRecorder({"squeue": "7 PREEMPTED\n", "sacct": "7 PREEMPTED\n"})
    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(slurm_module.subprocess, "run", recorder)

    states = SlurmJob.bulk_state([job])

    assert states[job] == "pending"
    # squeue answered for the id, so no sacct fallback happened.
    assert len(recorder.calls) == 1
    assert recorder.calls[0][0].endswith("squeue")
    # A non-terminal state must not finalize the (still-streaming) job log.
    cast("MagicMock", job.workspace.finalize_job_log).assert_not_called()


def test_slurm_bulk_state_out_of_queue_preempted_is_failed(monkeypatch, tmp_path) -> None:
    """A preempted job that has left the queue (e.g. PreemptMode=CANCEL) is a real failure."""
    job = _make_slurm_job(slurm_id="9", x=0)
    log = tmp_path / "j.log"
    log.write_text("streaming output")
    job.log_path = log

    # squeue no longer knows the job; sacct reports the terminal PREEMPTED.
    recorder = _RunRecorder({"squeue": "", "sacct": "9 PREEMPTED\n"})
    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(slurm_module.subprocess, "run", recorder)

    states = SlurmJob.bulk_state([job])

    assert states[job] == "failed"
    # Fell through squeue (empty) to sacct.
    assert len(recorder.calls) == 2
    assert recorder.calls[1][0].endswith("sacct")
    # A terminal state finalizes the log exactly once.
    cast("MagicMock", job.workspace.finalize_job_log).assert_called_once_with(log)


def test_slurm_bulk_state_preempt_requeue_transition_is_not_terminal(monkeypatch, tmp_path) -> None:
    """Walk a job through preempt -> requeue -> rerun -> completion.

    On a PreemptMode=REQUEUE cluster the same slurm id cycles through PREEMPTED
    (transient) and REQUEUED before running again. None of those intermediate
    states are terminal, so the job log must stay open until the rerun actually
    COMPLETEs -- a premature finalize would give up on the DAG mid-preemption.
    """
    job = _make_slurm_job(slurm_id="55", x=0)
    log = tmp_path / "j.log"
    log.write_text("streaming output")
    job.log_path = log

    monkeypatch.setattr(slurm_module, "_resolve_slurm_cmd", lambda name: f"/usr/bin/{name}")

    def poll(squeue_reply: str, sacct_reply: str = "") -> str:
        recorder = _RunRecorder({"squeue": squeue_reply, "sacct": sacct_reply})
        monkeypatch.setattr(slurm_module.subprocess, "run", recorder)
        return SlurmJob.bulk_state([job])[job]

    # The controller keeps the job the whole way through the preempt/requeue
    # cycle, so every intermediate poll is non-terminal.
    assert poll("55 RUNNING\n") == "running"
    assert poll("55 PREEMPTED\n") == "pending"  # caught mid-preemption / grace time
    assert poll("55 REQUEUED\n") == "pending"
    assert poll("55 PENDING\n") == "pending"
    assert poll("55 RUNNING\n") == "running"
    cast("MagicMock", job.workspace.finalize_job_log).assert_not_called()

    # The rerun finishes; squeue has dropped the id and sacct is authoritative.
    assert poll("", "55 COMPLETED\n") == "done"
    cast("MagicMock", job.workspace.finalize_job_log).assert_called_once_with(log)
