from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import is_dataclass
from typing import Any

from .executor import Executor
from .task import Task
from .utils.cached_property import cached_property
from .workspace import Workspace

# TODO: encourage implementing as frozen dataclass


class Experiment(ABC):
    def _is_frozen_dataclass(self) -> bool:
        return is_dataclass(self) and getattr(self, "__dataclass_params__").frozen

    @abstractmethod
    def tasks(self) -> dict[str, Task]:
        raise NotImplementedError

    def __getattribute__(self, name: str) -> Any:
        if name == "tasks" and self._is_frozen_dataclass():
            return cached_property(self, type(self).tasks, key="__cached_task_graph__")
        return super().__getattribute__(name)

    def __getitem__(self, item: str):
        return self.tasks()[item]

    def run(self, workspace: Workspace | None, executor: Executor | None) -> None:
        Task((lambda **kwargs: None), **self.tasks()).run(workspace=workspace, executor=executor).result()

    @classmethod
    def cli(cls):
        # build workspace, executor, experiment from CLI
        executor = Executor()  # Executor(executor args)
        workspace = Workspace()  # Workspace(workspace args)
        experiment = cls()  # cls(experiment args)
        experiment.run(workspace=workspace, executor=executor)
