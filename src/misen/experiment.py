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

    def _is_frozen_dataclass(self) -> bool:
        return is_dataclass(self) and getattr(self, "__dataclass_params__").frozen

    def get_step_graph(self) -> dict[str, Task]:
        if self._is_frozen_dataclass():
            if not hasattr(self, "__cached_graph__"):
                self.__cached_graph__ = self.step_graph
            return self.__cached_graph__

        return self.step_graph

    # TODO: inherit caching decorators for abstract methods
    # https://stackoverflow.com/a/57979319

    def run(self, executor: Executor, workspace: Workspace):
        ...
        # executor.submit(workspace=workspace)

    def result(self, workspace: Workspace, step_name: str): ...

    @classmethod
    def cli(cls):
        # build workspace, executor, experiment from CLI
        executor = Executor()  # Executor(executor args)
        workspace = Workspace()  # Workspace(workspace args)
        experiment = cls()  # cls(experiment args)
        experiment.run(executor, workspace)
