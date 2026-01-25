"""SLURM-backed executor implementation."""

from __future__ import annotations

import functools
import shlex
import shutil
import subprocess
from typing import TYPE_CHECKING, Literal

import cloudpickle

from misen.executor import Executor, Job, WorkUnit
from misen.utils.hashes import short_hash

if TYPE_CHECKING:
    from misen.workspace import Workspace


class SlurmJob(Job):
    """Job implementation backed by SLURM commands."""

    __slots__ = ("job_id",)

    def __init__(self, work_unit: WorkUnit, job_id: str) -> None:
        """Initialize a SLURM job wrapper."""
        super().__init__(work_unit=work_unit)
        self.job_id = job_id

    def state(self) -> Literal["pending", "running", "done", "failed", "unknown"]:
        """Return the job state based on SLURM CLI output."""
        state = _query_slurm_state(self.job_id)

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


class SlurmExecutor(Executor[SlurmJob]):
    """Executor implementation that submits work to SLURM."""

    def _dispatch(self, work_unit: WorkUnit, dependencies: set[SlurmJob], workspace: Workspace) -> SlurmJob:
        """Dispatch a work unit to SLURM via sbatch."""
        job_dir = (workspace.get_temp_dir() / "slurm").resolve()
        job_dir.mkdir(parents=True, exist_ok=True)

        payload_path = job_dir / f"misen-{short_hash(work_unit)}.pkl"  # TODO: more unique file name
        payload = cloudpickle.dumps(functools.partial(work_unit.execute, workspace=workspace))
        payload_path.write_bytes(payload)

        sbatch_cmd: list[str] = [SBATCH, "--parsable"]

        resources = work_unit.resources
        sbatch_cmd.extend(["--job-name", f"misen-{short_hash(work_unit)}"])
        sbatch_cmd.extend(["--ntasks-per-node", "1"])
        sbatch_cmd.extend(["--nodes", str(resources.nodes)])
        sbatch_cmd.extend(["--cpus-per-task", str(resources.cpus)])
        sbatch_cmd.extend(["--mem", f"{resources.memory}G"])
        sbatch_cmd.extend(["--gpus-per-node", str(resources.gpus)])
        sbatch_cmd.extend(["--time", str(resources.time or 1)])
        sbatch_cmd.extend(["--output", str(job_dir / "%j.out")])  # TODO: workspace provided?

        if dependencies:
            dep_ids = ":".join(job.job_id for job in dependencies)
            sbatch_cmd.extend(["--dependency", f"afterok:{dep_ids}"])

        wrap = shlex.join(
            [
                "python",
                "-c",
                (
                    "import cloudpickle; from pathlib import Path; "
                    f"cloudpickle.loads(Path({str(payload_path)!r}).read_bytes())()"
                ),
            ]
        )
        sbatch_cmd.extend(["--wrap", wrap])

        try:
            result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            msg = f"sbatch failed: {(e.stderr or e.stdout or '').strip()}"
            raise RuntimeError(msg) from e

        out = result.stdout.strip()
        job_id = out.split(";", 1)[0].split(None, 1)[0]
        if not job_id.isdigit():
            msg = f"Unexpected sbatch output: {out!r}"
            raise RuntimeError(msg)

        return SlurmJob(work_unit=work_unit, job_id=job_id)


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
