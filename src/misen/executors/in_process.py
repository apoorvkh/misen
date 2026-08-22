"""In-process executor implementation.

This backend executes the full submitted task DAG synchronously in the current
Python process (no subprocess spawn, no external scheduler).
"""

from __future__ import annotations

import logging
from operator import is_
from typing import TYPE_CHECKING

from misen.executor import CompletedJob, Executor
from misen.sentinels import DASK_CLIENT
from misen.tasks import Task
from misen.utils.graph import DependencyGraph
from misen.utils.snapshot import apply_env_files_temporarily, token_base32
from misen.utils.task_utils import build_task_dependency_graph
from misen.utils.work_unit import WorkUnit

if TYPE_CHECKING:
    from misen.utils.snapshot import ProjectSnapshot
    from misen.workspace import Workspace

logger = logging.getLogger(__name__)


class InProcessExecutor(Executor[CompletedJob]):
    """Executor that runs the full task DAG in dependency order in-process.

    Snapshots are structurally impossible here — tasks execute as live
    objects in this process, so there is no boundary at which pinned code
    or a materialized environment could apply. ``snapshot`` therefore
    defaults to ``False`` (and enabling it has no effect: execution never
    dispatches through the snapshot machinery).
    """

    snapshot: bool = False

    def __post_init__(self) -> None:
        """Warn when snapshotting is requested; it cannot apply in-process."""
        if self.snapshot:
            logger.warning(
                "InProcessExecutor ignores snapshot=True: tasks execute as live objects in "
                "this process, so pinned code and materialized environments cannot apply. "
                "Use LocalExecutor for snapshotted subprocess execution."
            )

    def submit(
        self,
        tasks: set[Task],
        workspace: Workspace,
        *,
        blocking: bool = False,
    ) -> DependencyGraph[CompletedJob]:
        """Execute submitted tasks synchronously in dependency order.

        Args:
            tasks: Root tasks requested by the caller.
            workspace: Workspace used for cache inspection and task execution.
            blocking: Unused for this executor because execution is already
                synchronous.

        Returns:
            Single-node job graph (or empty graph when no tasks were submitted).
        """
        _ = blocking
        logger.info("InProcessExecutor executing %d root task(s) synchronously.", len(tasks))
        job_id = token_base32(6)

        union = Task((lambda *_: None), *tasks)
        task_graph = build_task_dependency_graph(task=union)
        task_graph.remove_node_by_value(union, cmp=is_, first=True)
        if any(
            task.resources["nodes"] != 1 or task._requests_runtime_value(DASK_CLIENT)  # noqa: SLF001
            for task in task_graph.nodes()
        ):
            msg = "InProcessExecutor supports only single-node tasks and cannot provide DASK_CLIENT."
            raise ValueError(msg)

        with apply_env_files_temporarily():
            WorkUnit.execute(graph=task_graph, workspace=workspace, job_id=job_id)

        logger.info("InProcessExecutor finished executing %d task node(s).", len(list(task_graph.node_indices())))
        return DependencyGraph()

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[CompletedJob],
        workspace: Workspace,
        snapshot: ProjectSnapshot | None,
    ) -> CompletedJob:
        """Raise because this executor overrides :meth:`submit` directly."""
        _ = work_unit, dependencies, workspace, snapshot
        msg = "InProcessExecutor dispatches directly in submit()."
        raise RuntimeError(msg)
