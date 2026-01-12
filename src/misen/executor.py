"""
Executor interface for submitting a Task's DAG to an execution backend (e.g. a local background process or SLURM).

Convention: Dependency graph edges A -> B indicate that A depends on B.

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
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar, cast, get_args

from typing_extensions import assert_never

from .task import Task, TaskResources
from .utils.hashes import short_hash
from .utils.settings import FromSettingsABC

if TYPE_CHECKING:
    from .utils.graph import DependencyGraph
    from .workspace import Workspace

__all__ = ["Executor", "Job"]

ExecutorType: TypeAlias = Literal["local", "slurm"]
JobT = TypeVar("JobT", bound="Job")


class Executor(FromSettingsABC, Generic[JobT]):
    """Abstract interface for implementing an Executor for a specific backend."""

    def submit(
        self,
        tasks: set[Task],
        workspace: Workspace,
    ) -> DependencyGraph[CompletedJob | JobT]:
        """
        Submit a set of tasks for execution. Tasks will be run by backend respecting dependency order.

        Args:
            tasks: The set of tasks whose transitive dependencies define the Task DAG to run.
            workspace: Workspace providing Task artifact caching and retrieval.

        Returns:
            A dependency graph of backend-specific Job handles corresponding to WorkUnits.
        """
        work_graph: DependencyGraph[WorkUnit] = self._build_work_graph(tasks=tasks, workspace=workspace)

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
        """
        Dispatch a WorkUnit to the backend. Will run `work_unit.execute(workspace)` after dependencies are completed.

        Args:
            work_unit: The WorkUnit to dispatch.
            dependencies: Job handles corresponding to prerequisite (incomplete) WorkUnits.
            workspace: Workspace providing Task artifact caching and retrieval.

        Returns:
            A Job handle that can be queried for execution state.
        """

    @staticmethod
    def _build_work_graph(tasks: set[Task], workspace: Workspace) -> DependencyGraph[WorkUnit]:
        """
        Given a set of tasks, transform their Task DAG into a DAG of WorkUnits.
        """
        # Task dependency graph: an edge from A to B means A depends on B
        union = Task((lambda *_: None), *tasks)
        task_graph: DependencyGraph[Task] = union.dependency_graph(workspace=workspace)
        task_graph.remove_node_by_value(union, cmp=is_, first=True)

        # Retain only root and cachable tasks (and the induced graph minor)
        anchor_graph = task_graph.copy()
        anchor_graph.coarsen_to_anchors(
            anchors=[
                i for i in anchor_graph.node_indices() if anchor_graph.is_root(i) or anchor_graph[i].properties.cache
            ],
        )

        # replace nodes with WorkUnit instances
        work_graph = cast("DependencyGraph[WorkUnit]", anchor_graph.copy())
        for i in work_graph.evaluation_order():
            work_graph[i] = WorkUnit(root=anchor_graph[i], dependencies=set(work_graph.successors(i)))

        return work_graph

    """FromSettingsABC implementation. Permits initializing an Executor class from TOML settings or CLI."""

    @staticmethod
    def _settings_key() -> str:
        return "executor"

    @staticmethod
    def _default() -> Executor:
        from .executors.local import LocalExecutor

        return LocalExecutor()

    @classmethod
    def _resolve_type(cls, type_name: str | ExecutorType) -> type[Executor]:
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


class WorkUnit:
    """
    A WorkUnit is a cache-bounded unit of execution.

    It corresponds to the sub-DAG of non-cacheable tasks reachable from `root`, truncated at downstream cacheable tasks
    (which become separate WorkUnits, i.e. `dependencies`).

    Execution: Tasks in `graph` are executed sequentially in dependency order via `execute()`.

    Specifies minimum resources to execute any Task in the DAG.
    """

    root: Task
    graph: DependencyGraph[Task]
    resources: TaskResources
    dependencies: set[WorkUnit]

    def __init__(self, root: Task, dependencies: set[WorkUnit]) -> None:
        self.root = root
        self.dependencies = dependencies

        # Dependency graph of tasks.
        # Truncated at caching boundaries since downstream cacheable nodes become WorkUnit dependencies.
        self.graph = root.dependency_graph(exclude_cacheable=True)

        # Union of resources for all tasks in graph
        _resource_list: list[TaskResources] = [task.resources for task in self.graph.nodes()]
        self.resources = TaskResources(
            time=(
                None
                if any(r.time is None for r in _resource_list)
                else sum(cast("int", r.time) for r in _resource_list)
            ),
            nodes=max(r.nodes for r in _resource_list),
            memory=max(r.memory for r in _resource_list),
            cpus=max(r.cpus for r in _resource_list),
            gpus=max(r.gpus for r in _resource_list),
            gpu_memory=(
                None
                if all(r.gpu_memory is None for r in _resource_list)
                else max(r.gpu_memory for r in _resource_list if r.gpu_memory is not None)
            ),
        )

    def __hash__(self) -> int:
        """WorkUnits have a 1:1 correspondence with the `root` Task, so we can defer to hash(`root`)."""
        return hash(self.root)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WorkUnit) and self.root == other.root

    def __repr__(self) -> str:
        return f"WorkUnit(hash={short_hash(self)})"

    def execute(self, workspace: Workspace) -> None:
        """Execute self.graph Tasks one-by-one in dependency order. Should be called by `Executor._dispatch()`."""

        task_results: dict[Task, Any] = {}

        def resolve_arg(arg: Any) -> Any:
            if isinstance(arg, Task) and not arg.properties.cache:
                return task_results[arg]
            return arg

        ordered_tasks: list[Task] = list(self.graph)
        for i, task in enumerate(ordered_tasks):
            # only keep task results needed by future dependents
            remaining_deps = {d for t in ordered_tasks[i:] for d in t.dependencies}
            task_results = {k: v for k, v in task_results.items() if k in remaining_deps}

            # execute task
            # retrieving results of non-cacheable dependencies from `task_results`
            # cacheable dependencies will be resolved from Workspace in `.result()`
            task_results[task] = Task(
                task.func,
                *(resolve_arg(dep) for dep in task.args),
                **{k: resolve_arg(dep) for k, dep in task.kwargs.items()},
            ).result(
                workspace=workspace,
                compute_if_uncached=True,
                compute_uncached_deps=False,
            )


class Job(ABC):
    def __init__(self, work_unit: WorkUnit) -> None:
        self.work_unit = work_unit

    @abstractmethod
    def state(self) -> Literal["pending", "running", "done", "failed", "unknown"]: ...

    def wait(self, poll_s: float = 0.5) -> None:
        while True:
            if self.state() in ("done", "failed"):
                return
            time.sleep(poll_s)


class CompletedJob(Job):
    def __init__(self, work_unit: WorkUnit) -> None:
        super().__init__(work_unit=work_unit)

    def state(self) -> Literal["done"]:
        return "done"
