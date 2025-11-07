from pathlib import Path
from typing import Callable

import submitit

from ..executor import Executor, Job
from ..task import TaskResources


class SlurmJob(Job):
    def __init__(self, submitit_job: submitit.Job):
        self.submitit_job = submitit_job


class SlurmExecutor(Executor[SlurmJob]):
    def __init__(self, folder: Path = Path(".submitit")) -> None:
        self.slurm_executor = submitit.SlurmExecutor(folder=folder)

    def _submit(
        self,
        function: Callable,
        resources: TaskResources,
        dependencies: set[SlurmJob],
    ) -> SlurmJob:
        dependency = ",".join([f"afterok:{j.submitit_job.job_id}" for j in dependencies])

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

        job = self.slurm_executor.submit(function)

        return SlurmJob(submitit_job=job)
