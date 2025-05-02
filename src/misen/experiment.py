from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import is_dataclass
from typing import TYPE_CHECKING

from .executor import Executor
from .utils.cached_property import cached_property
from .workspace import Workspace

# if TYPE_CHECKING:
from .task import Task, task


# TODO: encourage implementing as frozen dataclass


class Experiment(ABC):
    def _is_frozen_dataclass(self) -> bool:
        return is_dataclass(self) and getattr(self, "__dataclass_params__").frozen

    @abstractmethod
    def step_graph(self) -> dict[str, Task]:
        raise NotImplementedError

    def __getattribute__(self, name):
        if name == "step_graph" and self._is_frozen_dataclass():
            return cached_property(self, type(self).step_graph, key="__cached_step_graph__")
        return super().__getattribute__(name)

    def run(self, executor: Executor, workspace: Workspace):
        # big brain??
        # this is literally all we need i think
        def meta_task(**kwargs) -> None:
            pass

        Task(meta_task, **self.step_graph()).run(workspace=workspace, executor=executor).result()

    # just pull from cache, basically
    def result(self, workspace: Workspace, step_name: str):
        return self.step_graph()[step_name].result(workspace)

    @classmethod
    def cli(cls):
        # build workspace, executor, experiment from CLI
        executor = Executor()  # Executor(executor args)
        workspace = Workspace()  # Workspace(workspace args)
        experiment = cls()  # cls(experiment args)
        experiment.run(executor, workspace)
