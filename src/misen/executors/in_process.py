"""In-process executor implementation.

This backend executes the full submitted task DAG synchronously in the current
Python process (no subprocess spawn, no external scheduler).
"""

from __future__ import annotations

from operator import is_
from typing import TYPE_CHECKING

from misen.executor import CompletedJob, Executor
from misen.tasks import Task
from misen.utils.graph import DependencyGraph
from misen.utils.snapshot import NullSnapshot, apply_env_files_temporarily
from misen.utils.work_unit import WorkUnit, build_task_dependency_graph

if TYPE_CHECKING:
    from misen.workspace import Workspace


class InProcessExecutor(Executor[CompletedJob, NullSnapshot]):
    """Executor that runs the full task DAG in dependency order in-process."""

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
        null_work_unit = WorkUnit(root=Task(lambda: None), dependencies=set())
        job_id, _, _ = self._make_snapshot(workspace=workspace).prepare_job(
            null_work_unit, workspace=workspace, assigned_resources_getter=lambda: None, gpu_runtime="cuda"
        )

        union = Task((lambda *_: None), *tasks)
        task_graph = build_task_dependency_graph(task=union)
        task_graph.remove_node_by_value(union, cmp=is_, first=True)

        with apply_env_files_temporarily():
            WorkUnit.execute(graph=task_graph, workspace=workspace, job_id=job_id, assigned_resources=None)

        return DependencyGraph()

    def _make_snapshot(self, workspace: Workspace) -> NullSnapshot:
        """Return placeholder snapshot (unused by this executor)."""
        _ = workspace
        return NullSnapshot()

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[CompletedJob],
        workspace: Workspace,
        snapshot: NullSnapshot,
    ) -> CompletedJob:
        """Raise because this executor overrides :meth:`submit` directly."""
        _ = work_unit, dependencies, workspace, snapshot
        msg = "InProcessExecutor dispatches directly in submit()."
        raise RuntimeError(msg)
