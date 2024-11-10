from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .task import Task
    from .workspace import Workspace


class Executor(ABC):
    def submit(self, task: Task, workspace: Workspace):
        raise NotImplementedError


# TODO: implement LocalExecutor that implements local / async multi-processing / multi-threading

# TODO: implement SlurmExecutor based on submitit
