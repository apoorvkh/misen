from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import pytest

import misen.executor as executor_module
from misen import Task, meta
from misen.executor import Executor, Job
from misen.utils.snapshot import Snapshot
from misen.workspaces.disk import DiskWorkspace

if TYPE_CHECKING:
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace


@meta(id="executor_lifecycle_task", cache=False)
def executor_lifecycle_task() -> int:
    return 1


class FakeSnapshot(Snapshot):
    def __init__(self) -> None:
        self.cleaned = False

    def cleanup(self) -> None:
        self.cleaned = True

    def prepare_job(self, *args: object, **kwargs: object) -> tuple[str, list[str], dict[str, str]]:
        return "fake-job", [], {}


class FailedJob(Job):
    def state(self) -> Literal["failed"]:
        return "failed"


class FailingExecutor(Executor[FailedJob, FakeSnapshot]):
    snapshots: ClassVar[list[FakeSnapshot]] = []

    def _make_snapshot(self, workspace: Workspace) -> FakeSnapshot:
        _ = workspace
        snapshot = FakeSnapshot()
        self.snapshots.append(snapshot)
        return snapshot

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[FailedJob],
        workspace: Workspace,
        snapshot: FakeSnapshot,
    ) -> FailedJob:
        _ = dependencies, workspace, snapshot
        return FailedJob(work_unit=work_unit)


class LocalSnapshotExecutor(Executor[FailedJob, FakeSnapshot]):
    def _make_snapshot(self, workspace: Workspace) -> FakeSnapshot:
        _ = workspace
        return FakeSnapshot()

    def _dispatch(
        self,
        work_unit: WorkUnit,
        dependencies: set[FailedJob],
        workspace: Workspace,
        snapshot: FakeSnapshot,
    ) -> FailedJob:
        _ = dependencies, workspace, snapshot
        return FailedJob(work_unit=work_unit)


def test_blocking_submit_raises_and_cleans_up_failed_jobs(tmp_path) -> None:
    FailingExecutor.snapshots.clear()
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    executor = FailingExecutor()

    with pytest.raises(RuntimeError, match="failed job"):
        executor.submit(tasks={Task(executor_lifecycle_task)}, workspace=workspace, blocking=True)

    assert len(FailingExecutor.snapshots) == 1
    assert FailingExecutor.snapshots[0].cleaned


def test_make_local_snapshot_returns_fresh_snapshot(monkeypatch, tmp_path) -> None:
    created: list[object] = []

    class FakeLocalSnapshot:
        def __init__(self, snapshots_dir: object) -> None:
            self.snapshots_dir = snapshots_dir
            created.append(self)

    monkeypatch.setattr(executor_module, "LocalSnapshot", FakeLocalSnapshot)
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))
    executor = LocalSnapshotExecutor()

    first = executor._make_local_snapshot(workspace)  # noqa: SLF001
    second = executor._make_local_snapshot(workspace)  # noqa: SLF001

    assert first is not second
    assert created == [first, second]
