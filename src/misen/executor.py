"""Executor interface for submitting a Task's DAG to an execution backend (e.g. a local process or SLURM).

Convention: Graph edge A -> B indicates that A depends on B.

Overview:
  1. The Task DAG is decomposed into WorkUnits (connected subgraphs), anchored at DAG roots and cacheable Tasks.
     Each WorkUnit contains the reachable subgraph of non-cacheable Tasks (truncated at downstream cacheable Tasks).
  2. WorkUnits are submitted to the backend for execution in dependency order.
     WorkUnits should assume their dependencies' roots are already cached (and can simply be retrieved) at runtime.
  3. Jobs are yielded and can be used to monitor the execution status of each WorkUnit.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from operator import is_
from typing import TYPE_CHECKING, Generic, Literal, TypeAlias, TypeVar, cast, get_args

from typing_extensions import assert_never

from .task import Task
from .utils.settings import FromSettingsABC
from .utils.work_unit import WorkUnit

if TYPE_CHECKING:
    from .utils.graph import DependencyGraph
    from .workspace import Workspace

__all__ = ["Executor", "Job"]

ExecutorType: TypeAlias = Literal["local", "slurm"]
JobT = TypeVar("JobT", bound="Job")


class Executor(FromSettingsABC, Generic[JobT]):
    """Abstract interface for implementing an Executor for a specific backend."""

    def submit(self, tasks: set[Task], workspace: Workspace) -> DependencyGraph[CompletedJob | JobT]:
        """Submit a set of tasks for execution. Tasks will be run by backend respecting dependency order.

        Returns:
            A dependency graph of backend-specific Job handles corresponding to WorkUnits.
        """
        work_graph: DependencyGraph[WorkUnit] = _build_work_graph(tasks=tasks, workspace=workspace)

        # dispatch work units and collect job handles

        jobs: dict[WorkUnit, CompletedJob | JobT] = {}

        for w in work_graph:  # dependency order
            if w.root.done(workspace=workspace):
                jobs[w] = CompletedJob(work_unit=w)
            else:
                dependencies = {jobs[d] for d in w.dependencies if not isinstance(jobs[d], CompletedJob)}
                jobs[w] = self._dispatch(work_unit=w, dependencies=dependencies, workspace=workspace)

        # return job graph corresponding to work graph

        job_graph = cast("DependencyGraph[CompletedJob | JobT]", work_graph.copy())
        for i in job_graph.node_indices():
            job_graph[i] = jobs[work_graph[i]]

        return job_graph

    @abstractmethod
    def _dispatch(self, work_unit: WorkUnit, dependencies: set[JobT], workspace: Workspace) -> JobT:
        """Dispatch a WorkUnit to the backend. Will run `work_unit.execute(workspace)` after dependencies are completed.

        Args:
            work_unit: The WorkUnit to dispatch.
            dependencies: Job handles corresponding to prerequisite (incomplete) WorkUnits.
            workspace: Workspace providing Task artifact caching and retrieval.

        Returns:
            A Job handle that can be queried for execution state.
        """

    # Below: FromSettingsABC implementation. Permits initializing an Executor class from TOML settings or CLI.

    @staticmethod
    def _settings_key() -> str:
        """Return the TOML settings key for executor configuration."""
        return "executor"

    @staticmethod
    def _default() -> Executor:
        """Return the default executor implementation."""
        from .executors.local import LocalExecutor

        return LocalExecutor()

    @classmethod
    def _resolve_type(cls, type_name: str | ExecutorType) -> type[Executor]:
        """Resolve an executor type name to a class."""
        if type_name in get_args(ExecutorType):
            type_name = cast("ExecutorType", type_name)
            match type_name:
                case "local":
                    from .executors.local import LocalExecutor

                    return LocalExecutor
                case "slurm":
                    from .executors.slurm import SlurmExecutor

                    return SlurmExecutor
                case _:
                    assert_never(type_name)
        return super()._resolve_type(type_name)


class Job(ABC):
    """Abstract job handle returned by an executor."""

    __slots__ = ("work_unit",)

    work_unit: WorkUnit

    def __init__(self, work_unit: WorkUnit) -> None:
        """Initialize the job wrapper for a work unit."""
        self.work_unit = work_unit

    @abstractmethod
    def state(self) -> Literal["pending", "running", "done", "failed", "unknown"]:
        """Return the current job state."""
        ...

    def wait(self, poll_s: float = 0.5) -> None:
        """Block until the job reaches a terminal state."""
        while True:
            if self.state() in ("done", "failed"):
                return
            time.sleep(poll_s)


class CompletedJob(Job):
    """Job placeholder for already-completed work units."""

    __slots__ = ()

    def __init__(self, work_unit: WorkUnit) -> None:
        """Initialize a completed job wrapper."""
        super().__init__(work_unit=work_unit)

    def state(self) -> Literal["done"]:
        """Return the completed state."""
        return "done"


def _build_work_graph(tasks: set[Task], workspace: Workspace) -> DependencyGraph[WorkUnit]:
    """Given a set of tasks, transform their Task DAG into a DAG of WorkUnits."""
    # Task dependency graph: an edge from A to B means A depends on B
    union = Task((lambda *_: None), *tasks)
    task_graph: DependencyGraph[Task] = union.dependency_graph(workspace=workspace)
    task_graph.remove_node_by_value(union, cmp=is_, first=True)

    # Retain only root and cachable tasks (and the induced graph minor)
    anchor_graph = task_graph.copy()
    anchors = [i for i in anchor_graph.node_indices() if anchor_graph.is_root(i) or anchor_graph[i].properties.cache]
    anchor_graph.coarsen_to_anchors(anchors=anchors)

    # replace nodes with WorkUnit instances
    work_graph = cast("DependencyGraph[WorkUnit]", anchor_graph.copy())
    for i in work_graph.evaluation_order():
        work_graph[i] = WorkUnit(root=anchor_graph[i], dependencies=set(work_graph.successors(i)))

    return work_graph
