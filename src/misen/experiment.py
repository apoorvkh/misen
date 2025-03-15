from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .executor import Executor
from .workspace import Workspace

if TYPE_CHECKING:
    from .task import Task


# TODO: can we make Experiment (and inherited classes) immutable and then cache step_graph?
# Or cache step_graph using (deterministic hash of "self") as key? invalidate cache if this hash changes?
# think about multi-threading; see @functools.cached_property notes


class Experiment(ABC):
    @property
    @abstractmethod
    def step_graph(self) -> Task:  # actually Graph
        raise NotImplementedError

    def run(self, executor: Executor, workspace: Workspace):
        executor.submit(task=self.step_graph, workspace=workspace)

    @classmethod
    def cli(cls):
        # build workspace, executor, experiment from CLI
        executor = Executor()  # Executor(executor args)
        workspace = Workspace()  # Workspace(workspace args)
        experiment = cls()  # cls(experiment args)
        experiment.run(executor, workspace)
