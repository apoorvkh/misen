from pathlib import Path
from typing import Literal

import submitit

from misen.executor import Executor, Job, WorkUnit
from misen.workspace import Workspace


class SlurmJob(Job):
    def __init__(self, submitit_job: submitit.Job) -> None:
        self.submitit_job = submitit_job

    def state(self) -> Literal["pending", "running", "done", "failed", "unknown"]:
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
    folder: str = ".submitit"

    def __post_init__(self) -> None:
        self.slurm_executor = submitit.SlurmExecutor(folder=Path(self.folder))

    def _dispatch(self, work_unit: WorkUnit, dependencies: set[SlurmJob], workspace: Workspace) -> SlurmJob:
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

        return SlurmJob(submitit_job=job)
