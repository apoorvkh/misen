from __future__ import annotations
from abc import ABCMeta, abstractmethod
import functools

from .core import Executor, Task, Workspace
from .utils import TaskGraphBuilder

class Experiment(metaclass=ABCMeta):

    @abstractmethod
    def calls(self):
        raise NotImplementedError

    # cached_property is not cloudpickle-able
    @functools.cached_property
    def step_graph(self):
        with TaskGraphBuilder(self.calls.__globals__):
            task_graph = self.calls()
            assert isinstance(task_graph, Task)
            return task_graph

    def run(self, executor: Executor, workspace: Workspace):
        executor.run(
            self.step_graph,
            workspace=workspace
        )

    @classmethod
    def cli(cls):
        # build workspace, executor, experiment from CLI
        executor = Executor()  # Executor(executor args)
        workspace = Workspace() # Workspace(workspace args)
        experiment = cls()  # cls(experiment args)
        experiment.run(executor, workspace)
