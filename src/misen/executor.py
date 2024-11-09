from __future__ import annotations

from abc import ABC
from . import Workspace, Task


class Executor(ABC):
    def run(self, task: Task, workspace: Workspace):
        raise NotImplementedError


class LocalExecutor(Executor):
    def run(self, task: Task, workspace: Workspace):
        if task in workspace:
            return workspace[task]

        return 
