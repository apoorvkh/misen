"""Executor graph-hook contracts shared by eager and workflow backends."""
# ruff: noqa: ANN001, D103, S101

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import pytest

import misen.executor as executor_module
import misen.utils.snapshot as snapshot_module
from misen import Task, meta
from misen.exceptions import SubmissionError
from misen.executor import CompletedJob, Executor, Job
from misen.utils.graph import DependencyGraph
from misen.utils.work_unit import WorkUnit
from misen.workspaces.disk import DiskWorkspace

if TYPE_CHECKING:
    from collections.abc import Sequence

    from misen.utils.snapshot import ProjectSnapshot
    from misen.workspace import Workspace


@meta(id="executor_graph_hook_task", cache=False)
def _graph_hook_task(value: int) -> int:
    return value


def _chain() -> tuple[DependencyGraph[WorkUnit], WorkUnit, WorkUnit, WorkUnit, tuple[int, int, int]]:
    cached = WorkUnit(root=Task(_graph_hook_task, value=1), dependencies=set())
    middle = WorkUnit(root=Task(_graph_hook_task, value=2), dependencies={cached})
    root = WorkUnit(root=Task(_graph_hook_task, value=3), dependencies={middle})

    graph: DependencyGraph[WorkUnit] = DependencyGraph()
    cached_index = graph.add_node(cached)
    middle_index = graph.add_node(middle)
    root_index = graph.add_node(root)
    graph.add_edge(middle_index, cached_index)
    graph.add_edge(root_index, middle_index)
    return graph, cached, middle, root, (cached_index, middle_index, root_index)


class _HookJob(Job):
    def state(self) -> Literal["pending"]:
        return "pending"


class _RejectingGraphExecutor(Executor[_HookJob]):
    validation_calls: ClassVar[list[tuple[DependencyGraph[WorkUnit], tuple[WorkUnit, ...], Workspace]]] = []

    def _validate_submission(
        self,
        *,
        work_graph: DependencyGraph[WorkUnit],
        pending_work_units: Sequence[WorkUnit],
        workspace: Workspace,
    ) -> None:
        self.validation_calls.append((work_graph, tuple(pending_work_units), workspace))
        msg = "unsupported remote graph"
        raise SubmissionError(msg)

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[_HookJob],
        workspace: Workspace,
        snapshot: ProjectSnapshot | None,
    ) -> _HookJob:
        del work_unit, dependencies, workspace, snapshot
        msg = "dispatch must not run after validation fails"
        raise AssertionError(msg)


class _RecordingEagerExecutor(Executor[_HookJob]):
    dispatches: ClassVar[list[tuple[WorkUnit, set[_HookJob]]]] = []

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[_HookJob],
        workspace: Workspace,
        snapshot: ProjectSnapshot | None,
    ) -> _HookJob:
        del workspace, snapshot
        self.dispatches.append((work_unit, dependencies))
        return _HookJob(work_unit=work_unit, job_id=f"job-{len(self.dispatches)}")


def test_backend_validation_runs_before_snapshot_creation(monkeypatch, tmp_path) -> None:
    graph, cached, middle, root, _ = _chain()
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    snapshot_attempted = False

    def unexpected_snapshot(**_kwargs: object) -> object:
        nonlocal snapshot_attempted
        snapshot_attempted = True
        msg = "snapshot creation must follow backend validation"
        raise AssertionError(msg)

    def never_done(self: WorkUnit, workspace: Workspace) -> bool:
        del self, workspace
        return False

    monkeypatch.setattr(executor_module, "build_work_graph", lambda **_kwargs: graph)
    monkeypatch.setattr(WorkUnit, "done", never_done)
    monkeypatch.setattr(snapshot_module, "ProjectSnapshot", unexpected_snapshot)
    _RejectingGraphExecutor.validation_calls.clear()
    executor = _RejectingGraphExecutor()

    with pytest.raises(SubmissionError, match="unsupported remote graph"):
        executor.submit(tasks={Task(_graph_hook_task, value=99)}, workspace=workspace)

    assert not snapshot_attempted
    assert _RejectingGraphExecutor.validation_calls == [(graph, (cached, middle, root), workspace)]


def test_default_graph_dispatch_remains_eager_and_dependency_aware(monkeypatch, tmp_path) -> None:
    graph, cached, middle, root, indices = _chain()
    cached_index, middle_index, root_index = indices
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))

    def only_cached_done(self: WorkUnit, workspace: Workspace) -> bool:
        del workspace
        return self is cached

    monkeypatch.setattr(executor_module, "build_work_graph", lambda **_kwargs: graph)
    monkeypatch.setattr(WorkUnit, "done", only_cached_done)
    monkeypatch.setattr(snapshot_module, "ProjectSnapshot", lambda **_kwargs: object())
    _RecordingEagerExecutor.dispatches.clear()

    job_graph = _RecordingEagerExecutor().submit(
        tasks={Task(_graph_hook_task, value=99)},
        workspace=workspace,
    )

    assert [work_unit for work_unit, _ in _RecordingEagerExecutor.dispatches] == [middle, root]
    assert _RecordingEagerExecutor.dispatches[0][1] == set()
    middle_job = job_graph[middle_index]
    root_job = job_graph[root_index]
    assert _RecordingEagerExecutor.dispatches[1][1] == {middle_job}

    assert isinstance(job_graph[cached_index], CompletedJob)
    assert isinstance(middle_job, _HookJob)
    assert isinstance(root_job, _HookJob)
    assert job_graph.successors(middle_index) == [job_graph[cached_index]]
    assert job_graph.successors(root_index) == [middle_job]
