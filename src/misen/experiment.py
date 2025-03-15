from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import is_dataclass
from typing import TYPE_CHECKING

from .executor import Executor
from .workspace import Workspace

if TYPE_CHECKING:
    from .task import Task


# TODO: encourage implementing as frozen dataclass

class Experiment(ABC):
    @property
    @abstractmethod
    def step_graph(self) -> dict[str, Task]:
        raise NotImplementedError

    def get_step_graph(self) -> dict[str, Task]:
        # if frozen dataclass, cache step_graph
        if is_dataclass(self) and getattr(self, "__dataclass_params__").frozen:
            if not hasattr(self, "__cached_graph__"):
                self.__cached_graph__ = self.step_graph
            return self.__cached_graph__
        else:
            return self.step_graph

    def run(self, executor: Executor, workspace: Workspace):
        executor.submit(task=self.get_step_graph(), workspace=workspace)

    @classmethod
    def cli(cls):
        # build workspace, executor, experiment from CLI
        executor = Executor()  # Executor(executor args)
        workspace = Workspace()  # Workspace(workspace args)
        experiment = cls()  # cls(experiment args)
        experiment.run(executor, workspace)
