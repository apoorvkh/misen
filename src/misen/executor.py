from __future__ import annotations

from abc import ABC
from .workspace import Workspace
from .task import Task


class Executor(ABC):
    def submit(self, task: Task, workspace: Workspace):
        raise NotImplementedError


class LocalExecutor(Executor):
    def submit(self, task: Task, workspace: Workspace):
        return
