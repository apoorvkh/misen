from pathlib import Path

import submitit

from ..executor import Executor
from ..task import Task
from ..workspace import Workspace


class SlurmExecutor(Executor):
    def __init__(self, folder: Path = Path(".submitit")) -> None:
        self.slurm_executor = submitit.SlurmExecutor(folder=folder)

    # TODO: figure out a unified interface

    def _submit(self, dependency_graph: dict[Task, set[Task]], workspace: Workspace) -> None:
        unmet_deps: dict[Task, set[Task]] = {t: d.copy() for t, d in dependency_graph.items()}
        submitted_job_id: dict[Task, str] = {}

        while len(unmet_deps) > 0:
            task = next(k for k, v in unmet_deps.items() if len(v) == 0)
            del unmet_deps[task]
            for t in unmet_deps.keys():
                unmet_deps[t].discard(task)

            dependency = ",".join(f"afterok:{submitted_job_id[d]}" for d in dependency_graph[task])

            # TODO: use union of task resources
            self.slurm_executor.update_parameters(
                stderr_to_stdout=True,
                use_srun=False,
                time=task.resources.time or 1,
                nodes=task.resources.nodes,
                ntasks_per_node=1,
                mem=f"{task.resources.memory}G",
                cpus_per_task=task.resources.cpus,
                gpus_per_node=task.resources.gpus,
                dependency=dependency,
            )

            # TODO: new implementation for task.result with multiprocessing

            job = self.slurm_executor.submit(
                task.result,
                workspace=workspace,
                compute_if_uncached=True,
                compute_uncached_deps=True,
            )

            submitted_job_id[task] = job.job_id
