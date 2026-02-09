"""SLURM-backed executor implementation."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from functools import cache
from typing import TYPE_CHECKING, Literal

from misen.executor import Executor, Job, WorkUnit
from misen.utils.snapshot import LocalSnapshot

if TYPE_CHECKING:
    from pathlib import Path

    from misen.workspace import Workspace


class SlurmJob(Job):
    """Job implementation backed by SLURM commands."""

    __slots__ = ("slurm_job_id",)

    def __init__(self, work_unit: WorkUnit, job_id: str, slurm_job_id: str, log_path: Path) -> None:
        """Initialize a SLURM job wrapper."""
        super().__init__(work_unit=work_unit, job_id=job_id, log_path=log_path)
        self.slurm_job_id = slurm_job_id

    def state(self) -> Literal["pending", "running", "done", "failed", "unknown"]:
        """Return the job state based on SLURM CLI output."""
        state = _query_slurm_state(self.slurm_job_id)

        if state is None:
            return "unknown"

        state = state.strip().upper()
        state = state.split("+", maxsplit=1)[0]
        state = state.split(":", maxsplit=1)[0]

        match state:
            case "PENDING" | "CONFIGURING" | "SUSPENDED" | "REQUEUED" | "REQUEUED_HOLD" | "STAGE_OUT":
                return "pending"
            case "RUNNING" | "COMPLETING":
                return "running"
            case (
                "BOOT_FAIL"
                | "CANCELLED"
                | "DEADLINE"
                | "FAILED"
                | "NODE_FAIL"
                | "OUT_OF_MEMORY"
                | "PREEMPTED"
                | "TIMEOUT"
                | "TIMEOUT_SIGNAL"
                | "SPECIAL_EXIT"
            ):
                return "failed"
            case "COMPLETED":
                return "done"
        return "unknown"


class SlurmExecutor(Executor[SlurmJob, LocalSnapshot]):
    """Executor implementation that submits work to SLURM."""

    @classmethod
    @cache
    def _cached_snapshot(cls, snapshots_dir: Path) -> LocalSnapshot:
        """Return a cached local snapshot instance for this executor class."""
        return LocalSnapshot(snapshots_dir=snapshots_dir)

    def _make_snapshot(self, workspace: Workspace) -> LocalSnapshot:
        """Create or reuse the SLURM executor snapshot."""
        snapshots_dir = (workspace.get_temp_dir() / "snapshots").resolve()
        return SlurmExecutor._cached_snapshot(snapshots_dir=snapshots_dir)

    def _dispatch(
        self, work_unit: WorkUnit, dependencies: set[SlurmJob], workspace: Workspace, snapshot: LocalSnapshot
    ) -> SlurmJob:
        """Dispatch a work unit to SLURM via sbatch."""
        work_unit_repr = work_unit.root.task_hash().short_b32()
        resources = work_unit.resources

        sbatch_cmd: list[str] = [SBATCH, "--parsable"]
        sbatch_cmd.extend(["--job-name", f"misen-{work_unit_repr}"])
        sbatch_cmd.extend(["--ntasks-per-node", "1"])
        sbatch_cmd.extend(["--nodes", str(resources.nodes)])
        sbatch_cmd.extend(["--cpus-per-task", str(resources.cpus)])
        sbatch_cmd.extend(["--mem", f"{resources.memory}G"])
        sbatch_cmd.extend(["--gpus-per-node", str(resources.gpus)])
        sbatch_cmd.extend(["--time", str(resources.time or 1)])

        # TODO: make these configurable
        sbatch_cmd.extend(["--account", "default"])
        sbatch_cmd.extend(["--partition", "batch"])

        if dependencies:
            dep_ids = ":".join(job.slurm_job_id for job in dependencies)
            sbatch_cmd.extend(["--dependency", f"afterok:{dep_ids}"])

        job_id, argv, env_overrides = snapshot.prepare_job(work_unit=work_unit, workspace=workspace)

        job_log_path = workspace.get_job_log_path(job_id=job_id)
        sbatch_cmd.extend(["--output", str(job_log_path)])

        env_prefix = ["env", *[f"{k}={v}" for k, v in env_overrides.items()]]
        sbatch_cmd.extend(["--export", "ALL"])
        sbatch_cmd.extend(["--wrap", shlex.join([*env_prefix, *argv])])

        try:
            result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            msg = f"sbatch failed: {(e.stderr or e.stdout or '').strip()}"
            raise RuntimeError(msg) from e

        out = result.stdout.strip()
        slurm_job_id = out.split(";", 1)[0].split(None, 1)[0]
        if not slurm_job_id.isdigit():
            msg = f"Unexpected sbatch output: {out!r}"
            raise RuntimeError(msg)
        return SlurmJob(work_unit=work_unit, job_id=job_id, slurm_job_id=slurm_job_id, log_path=job_log_path)


def _query_slurm_state(job_id: str) -> str | None:
    """Fetch the SLURM job state using squeue or sacct."""
    squeue = subprocess.run([SQUEUE, "-h", "-j", job_id, "-o", "%T"], check=False, capture_output=True, text=True)  # noqa: S603
    if squeue.returncode == 0:
        output = squeue.stdout.strip()
        if output:
            return output.splitlines()[0].strip()

    sacct = subprocess.run([SACCT, "-n", "-j", job_id, "--format=State"], check=False, capture_output=True, text=True)  # noqa: S603
    if sacct.returncode == 0:
        output = sacct.stdout.strip()
        if output:
            return output.splitlines()[0].strip().split()[0]
    return None


def _resolve_slurm_cmd(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        msg = f"Required command {name!r} not found on PATH. Is SLURM installed/loaded on this system?"
        raise FileNotFoundError(msg)
    return path


SQUEUE = _resolve_slurm_cmd("squeue")
SACCT = _resolve_slurm_cmd("sacct")
SBATCH = _resolve_slurm_cmd("sbatch")
