from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .executor import Executor
from .utils.task_graph_builder import TaskGraphBuilder
from .workspace import Workspace

if TYPE_CHECKING:
    from .task import Task


class Experiment(ABC):
    @abstractmethod
    def calls(self):
        raise NotImplementedError

    # cached_property is not cloudpickle-able
    @functools.cached_property
    def step_graph(self):
        with TaskGraphBuilder(self.calls.__globals__):
            task_graph: Task = self.calls()
            return task_graph

    def run(self, executor: Executor, workspace: Workspace):
        executor.submit(task=self.step_graph, workspace=workspace)

    @classmethod
    def cli(cls):
        # build workspace, executor, experiment from CLI
        executor = Executor()  # Executor(executor args)
        workspace = Workspace()  # Workspace(workspace args)
        experiment = cls()  # cls(experiment args)
        experiment.run(executor, workspace)
