from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .task import Task
    from .workspace import Workspace


class Executor(ABC):
    def computable_groups(self, task: Task, workspace: Workspace):
        """
        This function should partition (s.t. `task` and every cache=True Task is the root of a partition).
        Then we should return the graph of partitions. Each partition should represented as a topologically-sorted list of its Tasks.

        I think we can use the rustworkx library if want to use efficient graph algorithms.
        """
        # TODO: implement this here

    def submit(self, task: Task, workspace: Workspace):
        raise NotImplementedError


# TODO: implement LocalExecutor that implements local / async multi-processing / multi-threading

# TODO: implement SlurmExecutor based on submitit
