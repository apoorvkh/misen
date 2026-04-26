from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import pytest

import misen.executor as executor_module
from misen import Task, meta
from misen.executor import CompletedJob, Executor, Job, JobState, bulk_job_states
from misen.utils.snapshot import Snapshot
from misen.utils.work_unit import WorkUnit
from misen.workspaces.disk import DiskWorkspace

if TYPE_CHECKING:
    from collections.abc import Sequence

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


@meta(id="bulk_state_task", cache=False)
def bulk_state_task(x: int = 0) -> int:
    return x


class _CountingJob(Job):
    """Job that counts ``state()`` calls and records ``bulk_state`` group sizes."""

    bulk_calls: ClassVar[list[int]] = []

    def __init__(self, work_unit: WorkUnit, state_value: JobState) -> None:
        super().__init__(work_unit=work_unit)
        self._state_value = state_value
        self.state_calls = 0

    def state(self) -> JobState:
        self.state_calls += 1
        return self._state_value

    @classmethod
    def bulk_state(cls, jobs: Sequence[Job]) -> dict[Job, JobState]:
        cls.bulk_calls.append(len(jobs))
        return super().bulk_state(jobs)


class _BatchSlurmJob(Job):
    """Stand-in for SLURM-style jobs that answers many states from one batch."""

    queries: ClassVar[list[list[str]]] = []

    def __init__(self, work_unit: WorkUnit, slurm_id: str, state_value: JobState) -> None:
        super().__init__(work_unit=work_unit, job_id=slurm_id)
        self.slurm_id = slurm_id
        self._state_value = state_value

    def state(self) -> JobState:
        return type(self).bulk_state([self])[self]

    @classmethod
    def bulk_state(cls, jobs: Sequence[Job]) -> dict[Job, JobState]:
        ids = [j.slurm_id for j in jobs if isinstance(j, cls)]
        cls.queries.append(ids)
        return {j: j._state_value for j in jobs if isinstance(j, cls)}  # noqa: SLF001


def _wu(x: int) -> WorkUnit:
    return WorkUnit(root=Task(bulk_state_task, x=x), dependencies=set())


def test_bulk_job_states_groups_by_class_and_dispatches_once_per_class() -> None:
    _CountingJob.bulk_calls.clear()
    _BatchSlurmJob.queries.clear()

    counting_jobs = [
        _CountingJob(work_unit=_wu(i), state_value="running")
        for i in range(3)
    ]
    slurm_jobs = [
        _BatchSlurmJob(work_unit=_wu(100 + i), slurm_id=str(i), state_value="done")
        for i in range(4)
    ]
    completed_jobs = [
        CompletedJob(work_unit=_wu(200 + i)) for i in range(2)
    ]
    all_jobs: list[Job] = [*counting_jobs, *slurm_jobs, *completed_jobs]

    states = bulk_job_states(all_jobs)

    # One bulk_state dispatch per concrete class, regardless of group size.
    assert _CountingJob.bulk_calls == [3]
    assert _BatchSlurmJob.queries == [["0", "1", "2", "3"]]
    # CompletedJob group is reported as done without per-job state() calls.
    assert all(states[job] == "done" for job in completed_jobs)
    assert all(states[job] == "running" for job in counting_jobs)
    assert all(states[job] == "done" for job in slurm_jobs)
    # _CountingJob's default impl falls back to per-job state() — that's the
    # cost a backend pays when it doesn't override bulk_state.
    assert all(j.state_calls == 1 for j in counting_jobs)


def test_bulk_job_states_treats_known_query_errors_as_unknown() -> None:
    class _RaisingJob(Job):
        def state(self) -> JobState:
            return "running"

        @classmethod
        def bulk_state(cls, jobs: Sequence[Job]) -> dict[Job, JobState]:
            _ = jobs
            msg = "controller down"
            raise OSError(msg)

    jobs = [_RaisingJob(work_unit=_wu(900))]
    states = bulk_job_states(jobs)
    assert states[jobs[0]] == "unknown"


def test_bulk_job_states_normalizes_invalid_states_to_unknown() -> None:
    from typing import cast

    class _BadJob(Job):
        def state(self) -> JobState:
            return "running"

        @classmethod
        def bulk_state(cls, jobs: Sequence[Job]) -> dict[Job, JobState]:
            return {j: cast("JobState", "bogus") for j in jobs}

    jobs = [_BadJob(work_unit=_wu(901))]
    states = bulk_job_states(jobs)
    assert states[jobs[0]] == "unknown"


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
