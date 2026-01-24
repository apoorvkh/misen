from pathlib import Path
from typing import Literal

import submitit

from misen.executor import Executor, Job, WorkUnit
from misen.workspace import Workspace


class SlurmJob(Job):
    """Job implementation backed by submitit/SLURM."""

    __slots__ = ("submitit_job",)

    submitit_job: submitit.Job

    def __init__(self, work_unit: WorkUnit, submitit_job: submitit.Job) -> None:
        """Initialize a SLURM job wrapper."""
        super().__init__(work_unit=work_unit)
        self.submitit_job = submitit_job

    def state(self) -> Literal["pending", "running", "done", "failed", "unknown"]:
        """Return the job state based on submitit status."""
        match self.submitit_job.state:
            case "PENDING" | "SUSPENDED":
                return "pending"
            case (
                "BOOT_FAIL"
                | "CANCELLED"
                | "DEADLINE"
                | "FAILED"
                | "NODE_FAIL"
                | "OUT_OF_MEMORY"
                | "PREEMPTED"
                | "TIMEOUT"
            ):
                return "failed"
            case "RUNNING":
                return "running"
            case "COMPLETED":
                return "done"
        return "unknown"


class SlurmExecutor(Executor[SlurmJob]):
    """Executor implementation that submits work to SLURM."""

    folder: str = ".submitit"

    def __post_init__(self) -> None:
        """Initialize the submitit executor."""
        self.slurm_executor = submitit.SlurmExecutor(folder=Path(self.folder))

    def _dispatch(self, work_unit: WorkUnit, dependencies: set[SlurmJob], workspace: Workspace) -> SlurmJob:
        """Dispatch a work unit to SLURM via submitit."""
        dependency = ",".join([f"afterok:{j.submitit_job.job_id}" for j in dependencies])

        resources = work_unit.resources

        self.slurm_executor.update_parameters(
            stderr_to_stdout=True,
            use_srun=False,
            time=resources.time or 1,
            nodes=resources.nodes,
            ntasks_per_node=1,
            mem=f"{resources.memory}G",
            cpus_per_task=resources.cpus,
            gpus_per_node=resources.gpus,
            dependency=dependency,
        )

        job = self.slurm_executor.submit(work_unit.execute, workspace)

        return SlurmJob(work_unit=work_unit, submitit_job=job)
