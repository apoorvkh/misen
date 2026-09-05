"""SkyPilot executor tests using a hermetic in-process SDK fake."""
# ruff: noqa: ANN001, D103, PLR2004, S101, SLF001

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
import time
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, call

import msgspec
import pytest

import misen.executor as executor_module
import misen.executors.skypilot as skypilot_module
import misen.utils.snapshot as snapshot_module
from misen import DASK_CLIENT, Task, meta
from misen.exceptions import CacheError, ConfigError, ExecutionError, StatusQueryError, StorageError, SubmissionError
from misen.executors.skypilot import SkyPilotExecutor, SkyPilotJob
from misen.utils.graph import DependencyGraph
from misen.utils.hashing import stable_hash
from misen.utils.work_unit import WorkUnit
from misen.workspace import Workspace
from misen.workspaces.memory import InMemoryWorkspace

if TYPE_CHECKING:
    from collections.abc import Sequence

    from misen.executor import CompletedJob


class _ExtendedSkyPilotExecutor(SkyPilotExecutor):
    """Exercise durable-key compatibility for downstream executor subclasses."""

    scheduling_class: str = "standard"


@meta(id="skypilot_chain_task", cache=False)
def _chain_task(value: int) -> int:
    return value


@meta(
    id="skypilot_cpu_task",
    cache=False,
    resources={"cpus": 4, "memory": 32, "time": 17},
)
def _cpu_task() -> None:
    return None


@meta(
    id="skypilot_gpu_task",
    cache=False,
    resources={"cpus": 8, "memory": 64, "accelerators": 2, "accelerator_memory": 40},
)
def _gpu_task() -> None:
    return None


@meta(id="skypilot_multinode_task", cache=False, resources={"nodes": 2})
def _multinode_task() -> None:
    return None


@meta(id="skypilot_dask_task", cache=False, resources={"nodes": 2})
def _dask_task(_client: object) -> None:
    return None


class _FakeResources:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.validated = False

    def validate(self) -> None:
        self.validated = True


class _FakeTask:
    def __init__(self, **kwargs: object) -> None:
        self.name = kwargs["name"]
        self.run = kwargs["run"]
        self.num_nodes = kwargs["num_nodes"]
        self.resources = kwargs["resources"]
        self.api_server_access = kwargs["api_server_access"]


class _FakeRequestId(str):
    """Match SkyPilot 0.13's non-JSON-native RequestId string subclass."""


def _fake_sky(
    *,
    launch_results: Sequence[object] = (([42], None),),
    request_statuses: Sequence[object] | None = None,
    queue_records: Sequence[object] = (),
    api_server_local: bool = True,
    cancelled_request_ids: Sequence[str] | None = None,
) -> SimpleNamespace:
    requests: dict[str, object] = {f"launch-request-{index}": result for index, result in enumerate(launch_results)}
    statuses = request_statuses or ("SUCCEEDED",) * len(launch_results)
    request_states: dict[str, object] = {f"launch-request-{index}": status for index, status in enumerate(statuses)}
    launch_count = 0

    def launch(_task: object, *, name: str, pool: str | None) -> _FakeRequestId:
        nonlocal launch_count
        del name, pool
        request_id = _FakeRequestId(f"launch-request-{launch_count}")
        launch_count += 1
        return request_id

    jobs = SimpleNamespace(
        cancel=MagicMock(return_value="cancel-request"),
        launch=MagicMock(side_effect=launch),
        queue_v2=MagicMock(return_value="queue-request"),
    )
    cancelled = list(requests) if cancelled_request_ids is None else list(cancelled_request_ids)

    def get(request_id: str) -> object:
        if request_id == "api-cancel-request":
            return cancelled
        if request_id == "cancel-request":
            return None
        if request_id == "queue-request":
            return (list(queue_records), "ignored-metadata")
        if request_id in requests:
            result = requests[request_id]
            if isinstance(result, BaseException):
                raise result
            return result
        msg = f"Unexpected fake SkyPilot request: {request_id}"
        raise AssertionError(msg)

    def api_status(*, request_ids: Sequence[str]) -> list[object]:
        result = []
        for request_id in request_ids:
            if request_id not in request_states:
                continue
            status = request_states[request_id]
            record: dict[str, object] = {"request_id": request_id, "status": status}
            if str(status).rsplit(".", 1)[-1] in {"FAILED", "CANCELLED", "SUCCEEDED"}:
                record["finished_at"] = time.time() - skypilot_module._CANCELLED_REQUEST_RECONCILE_S - 1
            result.append(record)
        return result

    return SimpleNamespace(
        Task=_FakeTask,
        Resources=_FakeResources,
        api_cancel=MagicMock(return_value="api-cancel-request"),
        api_status=MagicMock(side_effect=api_status),
        jobs=jobs,
        get=MagicMock(side_effect=get),
        server=SimpleNamespace(
            common=SimpleNamespace(is_api_server_local=MagicMock(return_value=api_server_local)),
        ),
    )


def _remote_workspace() -> MagicMock:
    workspace = MagicMock(spec=Workspace)
    workspace.bootstrap_transport.return_value = "fetch-from-object-store"
    workspace.get_temp_dir.return_value = Path(".cache/misen/test-workspace")
    workspace.read_job_file.side_effect = FileNotFoundError
    workspace.get_result_hash.side_effect = CacheError("result is not committed")
    return workspace


def _work_unit(task: Task[Any]) -> WorkUnit:
    return WorkUnit(root=task, dependencies=set())


def test_skypilot_job_cancel_waits_for_native_request(tmp_path, monkeypatch) -> None:
    fake_sky = _fake_sky()
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=42,
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "job.log",
        workspace=cast("Workspace", _remote_workspace()),
    )

    job.cancel()

    fake_sky.jobs.cancel.assert_called_once_with(job_ids=[42])
    fake_sky.get.assert_called_once_with("cancel-request")


def test_skypilot_job_cancel_resolves_active_launch_before_cancelling(tmp_path, monkeypatch) -> None:
    fake_sky = _fake_sky(launch_results=(([73], None),))
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="launch-request-0",
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "job.log",
        workspace=cast("Workspace", _remote_workspace()),
    )

    job.cancel()

    assert job.managed_job_id == 73
    fake_sky.api_cancel.assert_not_called()
    fake_sky.jobs.cancel.assert_called_once_with(job_ids=[73])
    assert fake_sky.get.call_args_list == [call("launch-request-0"), call("cancel-request")]


def test_skypilot_job_cancel_cancels_before_persisting_resolved_id(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    workspace.read_job_file.side_effect = None
    workspace.read_job_file.return_value = msgspec.json.encode(
        executor_module._JobRecord(
            "local-1",
            "launch-request-0",
            "STATUSABC",
            60,
            request_id="launch-request-0",
        )
    )
    events: list[str] = []
    persistence_error = "object store unavailable"

    def fail_persistence(*_args: object) -> None:
        events.append("persist")
        raise StorageError(persistence_error)

    workspace.put_job_file.side_effect = fail_persistence
    fake_sky = _fake_sky(
        launch_results=(([73], None),),
    )
    original_get = fake_sky.get.side_effect

    def get_with_events(request_id: str) -> object:
        if request_id == "cancel-request":
            events.append("cancel")
        return original_get(request_id)

    fake_sky.get.side_effect = get_with_events
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="launch-request-0",
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "job.log",
        workspace=cast("Workspace", workspace),
    )
    job._bind_record(cast("Workspace", workspace), "record-key")

    with pytest.raises(ExecutionError, match="Could not persist managed-job ID"):
        job.cancel()

    assert job.managed_job_id == 73
    assert events == ["cancel", "persist"]
    fake_sky.api_cancel.assert_not_called()
    fake_sky.jobs.cancel.assert_called_once_with(job_ids=[73])

    def persist_on_retry(*_args: object) -> None:
        events.append("persist-retry")

    workspace.put_job_file.side_effect = persist_on_retry
    job.cancel()

    assert events == ["cancel", "persist", "cancel", "persist-retry"]
    assert job._managed_job_id_persisted
    assert fake_sky.jobs.cancel.call_count == 2


def test_skypilot_job_cancel_recovers_assigned_job_after_launch_result_loss(tmp_path, monkeypatch) -> None:
    managed_name = "misen-statusabc-local-1"
    fake_sky = _fake_sky(
        launch_results=(RuntimeError("launch response unavailable"),),
        queue_records=({"job_id": 73, "task_id": 0, "job_name": managed_name},),
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="launch-request-0",
        managed_job_name=managed_name,
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "job.log",
        workspace=cast("Workspace", _remote_workspace()),
    )

    job.cancel()

    assert job.managed_job_id == 73
    fake_sky.jobs.queue_v2.assert_called_once_with(
        refresh=True,
        fields=("job_id", "task_id", "job_name"),
    )
    fake_sky.jobs.cancel.assert_called_once_with(job_ids=[73])
    assert fake_sky.get.call_args_list == [
        call("launch-request-0"),
        call("queue-request"),
        call("cancel-request"),
    ]


def test_externally_cancelled_launch_publishes_gate_then_converges(tmp_path, monkeypatch) -> None:
    managed_name = "misen-statusabc-local-1"
    fake_sky = _fake_sky(request_statuses=("CANCELLED",), queue_records=())
    request_record = {
        "request_id": "launch-request-0",
        "status": "CANCELLED",
        "finished_at": time.time(),
    }
    fake_sky.api_status.side_effect = None
    fake_sky.api_status.return_value = [request_record]
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    workspace = _remote_workspace()
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="launch-request-0",
        managed_job_name=managed_name,
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "job.log",
        workspace=cast("Workspace", workspace),
    )

    assert SkyPilotJob.bulk_state([job]) == {job: "pending"}
    assert job._terminal_state is None
    workspace.put_job_file.assert_called_once_with("STATUSABC", "local-1.state", b"failed")

    workspace.read_job_file.side_effect = None
    workspace.read_job_file.return_value = b"failed"
    request_record["finished_at"] = time.time() - skypilot_module._CANCELLED_REQUEST_RECONCILE_S - 1

    assert SkyPilotJob.bulk_state([job]) == {job: "failed"}

    assert job.managed_job_id is None
    fake_sky.jobs.cancel.assert_not_called()


def test_externally_cancelled_launch_cancels_id_appearing_during_reconciliation(tmp_path, monkeypatch) -> None:
    managed_name = "misen-statusabc-local-1"
    fake_sky = _fake_sky(request_statuses=("CANCELLED",))
    fake_sky.api_status.side_effect = None
    fake_sky.api_status.return_value = [
        {"request_id": "launch-request-0", "status": "CANCELLED", "finished_at": time.time()}
    ]
    original_get = fake_sky.get.side_effect
    queue_results = iter(
        (
            ([], "ignored-metadata"),
            ([{"job_id": 73, "task_id": 0, "job_name": managed_name}], "ignored-metadata"),
            ([{"job_id": 73, "task_id": 0, "status": "CANCELLED"}], "ignored-metadata"),
        )
    )

    def get_with_late_job(request_id: str) -> object:
        if request_id == "queue-request":
            return next(queue_results)
        return original_get(request_id)

    fake_sky.get.side_effect = get_with_late_job
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    workspace = _remote_workspace()
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="launch-request-0",
        managed_job_name=managed_name,
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "job.log",
        workspace=cast("Workspace", workspace),
    )

    assert SkyPilotJob.bulk_state([job]) == {job: "pending"}
    workspace.read_job_file.side_effect = None
    workspace.read_job_file.return_value = b"failed"

    assert SkyPilotJob.bulk_state([job]) == {job: "failed"}
    assert job.managed_job_id == 73
    fake_sky.jobs.cancel.assert_called_once_with(job_ids=[73])


def test_skypilot_job_cancel_reports_inconclusive_name_recovery(tmp_path, monkeypatch) -> None:
    fake_sky = _fake_sky(launch_results=(RuntimeError("launch response unavailable"),))
    original_get = fake_sky.get.side_effect
    error_message = "jobs controller unavailable"

    def get_with_queue_failure(request_id: str) -> object:
        if request_id == "queue-request":
            raise RuntimeError(error_message)
        return original_get(request_id)

    fake_sky.get.side_effect = get_with_queue_failure
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="launch-request-0",
        managed_job_name="misen-statusabc-local-1",
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "job.log",
        workspace=cast("Workspace", _remote_workspace()),
    )

    with pytest.raises(ExecutionError, match="Could not recover SkyPilot managed jobs by name"):
        job.cancel()

    fake_sky.jobs.cancel.assert_not_called()


def test_skypilot_job_cancel_accepts_failed_launch_without_managed_job(tmp_path, monkeypatch) -> None:
    fake_sky = _fake_sky(
        launch_results=(RuntimeError("launch rejected"),),
        request_statuses=("FAILED",),
        queue_records=(),
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="launch-request-0",
        managed_job_name="misen-statusabc-local-1",
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "job.log",
        workspace=cast("Workspace", _remote_workspace()),
    )

    job.cancel()

    assert job.managed_job_id is None
    fake_sky.jobs.cancel.assert_not_called()
    fake_sky.api_status.assert_called_once_with(request_ids=["launch-request-0"])


def _diamond_graph() -> tuple[DependencyGraph[WorkUnit], tuple[WorkUnit, WorkUnit, WorkUnit, WorkUnit]]:
    base = _work_unit(Task(_chain_task, value=1))
    left = WorkUnit(root=Task(_chain_task, value=2), dependencies={base})
    right = WorkUnit(root=Task(_chain_task, value=3), dependencies={base})
    root = WorkUnit(root=Task(_chain_task, value=4), dependencies={left, right})
    graph: DependencyGraph[WorkUnit] = DependencyGraph()
    base_index = graph.add_node(base)
    left_index = graph.add_node(left)
    right_index = graph.add_node(right)
    root_index = graph.add_node(root)
    graph.add_edge(left_index, base_index)
    graph.add_edge(right_index, base_index)
    graph.add_edge(root_index, left_index)
    graph.add_edge(root_index, right_index)
    return graph, (base, left, right, root)


def test_submit_accepts_diamond_dag_and_launches_one_managed_job_per_work_unit(monkeypatch, tmp_path) -> None:
    graph, work_units = _diamond_graph()
    base, left, right, root = work_units
    workspace = _remote_workspace()
    fake_sky = _fake_sky(
        launch_results=tuple(([managed_id], {"name": "ignored"}) for managed_id in (311, 312, 313, 314))
    )
    prepared: list[tuple[WorkUnit, dict[WorkUnit, tuple[str, str]]]] = []

    class _FakeSnapshot:
        submission_id = "SUBMISSIONABC"

        def __init__(self, **_kwargs: object) -> None:
            pass

        def prepare_job(
            self,
            *,
            work_unit: WorkUnit,
            workspace: Workspace,
            dependency_jobs: dict[WorkUnit, tuple[str, str]],
        ) -> tuple[str, list[str], dict[str, str], Path]:
            del workspace
            prepared.append((work_unit, dependency_jobs))
            task_id = work_unit.root.kwargs["value"]
            return (
                f"misen-job-{task_id}",
                ["python", "-m", "misen.worker", str(task_id)],
                {"MISEN_TEST_VALUE": f"value {task_id}"},
                tmp_path / "logs" / f"{task_id}.log",
            )

    monkeypatch.setattr(executor_module, "build_work_graph", lambda **_kwargs: graph)

    def never_done(self: WorkUnit, workspace: Workspace) -> bool:
        del self, workspace
        return False

    monkeypatch.setattr(WorkUnit, "done", never_done)
    monkeypatch.setattr(snapshot_module, "ProjectSnapshot", _FakeSnapshot)
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)

    job_graph = SkyPilotExecutor(infra="gcp/us-central1").submit(
        tasks={Task(_chain_task, value=99)},
        workspace=cast("Workspace", workspace),
    )

    assert prepared == [
        (base, {}),
        (left, {base: ("SUBMISSIONABC", "misen-job-1")}),
        (right, {base: ("SUBMISSIONABC", "misen-job-1")}),
        (root, {left: ("SUBMISSIONABC", "misen-job-2"), right: ("SUBMISSIONABC", "misen-job-3")}),
    ]
    launch_calls = fake_sky.jobs.launch.call_args_list
    assert len(launch_calls) == 4
    sky_tasks = [cast("_FakeTask", launch_call.args[0]) for launch_call in launch_calls]
    assert all(task.num_nodes == 1 for task in sky_tasks)
    assert all(task.api_server_access is False for task in sky_tasks)
    assert all(
        launch_call.kwargs["name"] == task.name for launch_call, task in zip(launch_calls, sky_tasks, strict=True)
    )
    assert all(launch_call.kwargs["pool"] is None for launch_call in launch_calls)
    first_run = cast("str", sky_tasks[0].run)
    assert "timeout --signal=TERM --kill-after=30s 60m" in first_run
    assert "tee -a" in first_run
    assert "OMP_DYNAMIC=FALSE" in first_run
    assert "MKL_DYNAMIC=FALSE" in first_run
    assert "OPENBLAS_DYNAMIC=0" in first_run
    assert "OMP_NUM_THREADS=" not in first_run
    assert "CUDA_VISIBLE_DEVICES=" not in first_run

    jobs = [cast("SkyPilotJob", job_graph[index]) for index in range(4)]
    assert all(isinstance(job, SkyPilotJob) for job in jobs)
    assert [(job.request_id, job.job_id) for job in jobs] == [
        ("launch-request-0", "misen-job-1"),
        ("launch-request-1", "misen-job-2"),
        ("launch-request-2", "misen-job-3"),
        ("launch-request-3", "misen-job-4"),
    ]
    assert all(job.managed_job_id is None for job in jobs)
    fake_sky.get.assert_not_called()
    assert all(not hasattr(job, "pipeline_task_id") for job in jobs)
    assert [job.deadline_minutes for job in jobs] == [60, 120, 120, 180]
    assert job_graph.successors(1) == [jobs[0]]
    assert job_graph.successors(2) == [jobs[0]]
    assert set(job_graph.successors(3)) == {jobs[1], jobs[2]}


def test_independent_work_launches_without_dependency_gates(monkeypatch, tmp_path) -> None:
    first = _work_unit(Task(_chain_task, value=1))
    second = _work_unit(Task(_chain_task, value=2))
    fake_sky = _fake_sky(launch_results=(([41], None), ([42], None)))
    snapshot = SimpleNamespace(
        submission_id="PARALLELABC",
        prepare_job=MagicMock(
            side_effect=[
                ("job-1", ["python", "worker.py", "1"], {}, tmp_path / "one.log"),
                ("job-2", ["python", "worker.py", "2"], {}, tmp_path / "two.log"),
            ]
        ),
    )
    workspace = _remote_workspace()
    jobs: dict[WorkUnit, CompletedJob | SkyPilotJob] = {}
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)

    SkyPilotExecutor()._dispatch_work_graph(
        pending_work_units=[first, second],
        jobs=jobs,
        workspace=cast("Workspace", workspace),
        snapshot=cast("Any", snapshot),
        progress=MagicMock(),
    )

    assert [item.kwargs["dependency_jobs"] for item in snapshot.prepare_job.call_args_list] == [{}, {}]
    assert len(fake_sky.jobs.launch.call_args_list) == 2
    assert set(jobs) == {first, second}


def test_pool_is_forwarded_to_managed_job_launch(monkeypatch, tmp_path) -> None:
    work_unit = _work_unit(Task(_cpu_task))
    workspace = _remote_workspace()
    fake_sky = _fake_sky(launch_results=(([42], None),))
    snapshot = SimpleNamespace(
        submission_id="POOLEDABC",
        prepare_job=MagicMock(return_value=("job-1", ["python", "worker.py"], {}, tmp_path / "pool.log")),
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)

    job = SkyPilotExecutor(pool="misen-dev")._dispatch(
        work_unit=work_unit,
        dependencies=set(),
        workspace=cast("Workspace", workspace),
        snapshot=cast("Any", snapshot),
    )

    assert job.managed_job_id is None
    assert job.request_id == "launch-request-0"
    assert type(job.request_id) is str
    msgspec.json.encode(job._record())
    assert fake_sky.jobs.launch.call_args.kwargs["pool"] == "misen-dev"
    fake_sky.get.assert_not_called()


def test_unset_pool_preserves_pre_pool_durable_key_identity() -> None:
    executor = SkyPilotExecutor()

    # Pinned from the same default SkyPilotExecutor before ``pool`` became a
    # declared field; active unpooled jobs must remain reattachable on upgrade.
    assert stable_hash(executor._job_key_identity()) == 4424697057611365518

    pooled = SkyPilotExecutor(pool="misen-dev")
    assert pooled._job_key_identity().pool == "misen-dev"
    assert "manage_api_server" not in pooled._job_key_identity().__struct_fields__


def test_unset_pool_preserves_runtime_subclass_identity_and_fields() -> None:
    executor = _ExtendedSkyPilotExecutor(scheduling_class="priority")

    identity = executor._job_key_identity()
    identity_type = type(identity)
    assert identity_type.__module__ == type(executor).__module__
    assert identity_type.__name__ == type(executor).__name__
    assert identity_type.__qualname__ == type(executor).__qualname__
    assert identity_type.__struct_fields__ == tuple(
        field for field in executor.__struct_fields__ if field not in {"pool", "manage_api_server"}
    )
    assert identity.scheduling_class == "priority"

    changed = _ExtendedSkyPilotExecutor(scheduling_class="batch")._job_key_identity()
    assert type(changed) is identity_type
    assert stable_hash(changed) != stable_hash(identity)
    assert stable_hash(identity) != stable_hash(SkyPilotExecutor()._job_key_identity())

    pooled = _ExtendedSkyPilotExecutor(pool="misen-dev", scheduling_class="priority")
    assert pooled._job_key_identity().pool == "misen-dev"
    assert pooled._job_key_identity().scheduling_class == "priority"


@pytest.mark.parametrize("pool", ["misen-dev", "Research_Pool.2", "a"])
def test_skypilot_accepts_valid_pool_names(pool: str) -> None:
    assert SkyPilotExecutor(pool=pool).pool == pool


@pytest.mark.parametrize("pool", ["", "1pool", "-pool", "pool-", "pool/name", "pool name"])
def test_skypilot_rejects_invalid_pool_names(pool: str) -> None:
    with pytest.raises(ValueError, match="pool must start with a letter"):
        SkyPilotExecutor(pool=pool)


def test_pool_rejects_dependencies_between_pending_work_units() -> None:
    graph, work_units = _diamond_graph()

    with pytest.raises(ConfigError, match="dependency-independent pending work units"):
        SkyPilotExecutor(pool="misen-dev")._validate_submission(
            work_graph=graph,
            pending_work_units=list(work_units),
            workspace=cast("Workspace", _remote_workspace()),
        )


def test_submit_reattaches_live_parent_across_submission_namespaces(monkeypatch, tmp_path) -> None:
    parent = _work_unit(Task(_chain_task, value=1))
    child = WorkUnit(root=Task(_chain_task, value=2), dependencies={parent})
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    fake_sky = _fake_sky(
        launch_results=(([41], None), ([42], None)),
        queue_records=({"job_id": 41, "task_id": 0, "status": "RUNNING"},),
    )
    old_snapshot = SimpleNamespace(
        submission_id="OLD",
        snapshot_key="SNAPSHOT",
        prepare_job=MagicMock(return_value=("parent-job", ["python", "worker.py"], {}, tmp_path / "parent.log")),
    )
    new_snapshot = SimpleNamespace(
        submission_id="NEW",
        snapshot_key="SNAPSHOT",
        prepare_job=MagicMock(return_value=("child-job", ["python", "worker.py"], {}, tmp_path / "child.log")),
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    executor = SkyPilotExecutor()

    executor._dispatch_work_graph(
        pending_work_units=[parent], jobs={}, workspace=workspace, snapshot=old_snapshot, progress=MagicMock()
    )
    jobs: dict[Any, Any] = {}
    executor._dispatch_work_graph(
        pending_work_units=[parent, child], jobs=jobs, workspace=workspace, snapshot=new_snapshot, progress=MagicMock()
    )

    assert cast("SkyPilotJob", jobs[parent]).managed_job_id == 41
    assert new_snapshot.prepare_job.call_args.kwargs["dependency_jobs"] == {parent: ("OLD", "parent-job")}
    assert fake_sky.jobs.launch.call_count == 2


def test_partial_launch_error_reports_already_accepted_managed_jobs(monkeypatch, tmp_path) -> None:
    first = _work_unit(Task(_chain_task, value=1))
    second = _work_unit(Task(_chain_task, value=2))
    workspace = _remote_workspace()
    fake_sky = _fake_sky(launch_results=(([101], None),))
    launch = fake_sky.jobs.launch

    def fail_second_launch(task: object, *, name: str, pool: str | None) -> str:
        del task, pool
        if launch.call_count == 1:
            return "launch-request-0"
        msg = f"provider rejected {name}"
        raise RuntimeError(msg)

    launch.side_effect = fail_second_launch
    snapshot = SimpleNamespace(
        submission_id="PARTIALABC",
        prepare_job=MagicMock(
            side_effect=[
                ("job-1", ["python", "worker.py", "1"], {}, tmp_path / "one.log"),
                ("job-2", ["python", "worker.py", "2"], {}, tmp_path / "two.log"),
            ]
        ),
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)

    with pytest.raises(SubmissionError, match="after 1 earlier job") as exc_info:
        SkyPilotExecutor()._dispatch_work_graph(
            pending_work_units=[first, second],
            jobs={},
            workspace=cast("Workspace", workspace),
            snapshot=cast("Any", snapshot),
            progress=MagicMock(),
        )

    assert len(exc_info.value.submitted_jobs) == 1
    submitted = cast("SkyPilotJob", exc_info.value.submitted_jobs[0])
    assert submitted.work_unit is first
    assert submitted.managed_job_id is None
    assert submitted.request_id == "launch-request-0"


def test_cpu_resources_expand_to_ordered_infrastructure_alternatives() -> None:
    fake_sky = _fake_sky()
    executor = SkyPilotExecutor(
        infra=["aws", "gcp/us-central1"],
        instance_type="worker-shape",
        use_spot=True,
        image_id="image-tag",
        disk_size=200,
        max_hourly_cost=3.5,
        job_recovery="FAILOVER",
    )

    options = executor._resource_options(fake_sky, _work_unit(Task(_cpu_task)))

    assert isinstance(options, list)
    assert [option.kwargs for option in options] == [
        {
            "infra": "aws",
            "cpus": "4+",
            "memory": "32+",
            "use_spot": True,
            "instance_type": "worker-shape",
            "image_id": "image-tag",
            "disk_size": 200,
            "max_hourly_cost": 3.5,
            "job_recovery": "FAILOVER",
        },
        {
            "infra": "gcp/us-central1",
            "cpus": "4+",
            "memory": "32+",
            "use_spot": True,
            "instance_type": "worker-shape",
            "image_id": "image-tag",
            "disk_size": 200,
            "max_hourly_cost": 3.5,
            "job_recovery": "FAILOVER",
        },
    ]


@pytest.mark.parametrize(
    "infra",
    [
        pytest.param("azure/eastus", id="azure"),
        pytest.param("k8s/platform/research-cluster", id="kubernetes-context"),
        pytest.param("ssh/on-prem-gpu-pool", id="ssh-node-pool"),
        pytest.param("slurm/research-cluster", id="slurm-cluster"),
        pytest.param("oci/us-ashburn-1", id="oci"),
    ],
)
def test_resource_options_forward_skypilot_compatible_infrastructures(infra: str) -> None:
    option = SkyPilotExecutor(infra=infra)._resource_options(_fake_sky(), _work_unit(Task(_cpu_task)))

    assert isinstance(option, _FakeResources)
    assert option.kwargs == {
        "infra": infra,
        "cpus": "4+",
        "memory": "32+",
        "use_spot": False,
    }
    assert option.validated


def test_resource_options_preserve_order_across_backend_families() -> None:
    infras = [
        "azure/eastus",
        "k8s/platform/research-cluster",
        "ssh/on-prem-gpu-pool",
        "slurm/research-cluster",
        "oci/us-ashburn-1",
    ]

    options = SkyPilotExecutor(infra=infras)._resource_options(_fake_sky(), _work_unit(Task(_cpu_task)))

    assert isinstance(options, list)
    assert [option.kwargs["infra"] for option in options] == infras
    assert all(option.validated for option in options)


def test_remote_api_server_owns_environment_dependent_resource_validation() -> None:
    fake_sky = _fake_sky(api_server_local=False)

    option = SkyPilotExecutor(infra="ssh/on-prem-gpu-pool")._resource_options(
        fake_sky,
        _work_unit(Task(_cpu_task)),
    )

    assert isinstance(option, _FakeResources)
    assert not option.validated
    fake_sky.server.common.is_api_server_local.assert_called_once_with()


def test_gpu_resources_require_explicit_models_and_filter_by_device_memory() -> None:
    fake_sky = _fake_sky()
    work_unit = _work_unit(Task(_gpu_task))
    executor = SkyPilotExecutor(
        accelerators={"cuda": ["A100", "L4"]},
        accelerator_memory={"A100": 80, "L4": 24},
    )

    option = executor._resource_options(fake_sky, work_unit)

    assert isinstance(option, _FakeResources)
    assert option.kwargs == {
        "infra": "aws",
        "cpus": "8+",
        "memory": "64+",
        "use_spot": False,
        "accelerators": {"A100": 2},
    }

    with pytest.raises(SubmissionError, match="No SkyPilot accelerator models"):
        SkyPilotExecutor()._resource_options(fake_sky, work_unit)
    with pytest.raises(SubmissionError, match="minimum 40 GiB/device"):
        SkyPilotExecutor(accelerators={"cuda": ["L4"]})._resource_options(fake_sky, work_unit)


def test_invalid_sdk_resources_are_rejected_during_preflight(monkeypatch) -> None:
    work_unit = _work_unit(Task(_cpu_task))
    graph: DependencyGraph[WorkUnit] = DependencyGraph()
    graph.add_node(work_unit)
    fake_sky = _fake_sky()

    class _ValidatingResources(_FakeResources):
        def validate(self) -> None:
            if self.kwargs.get("job_recovery") == "RESTART":
                msg = "Invalid job recovery strategy: RESTART"
                raise ValueError(msg)

    fake_sky.Resources = _ValidatingResources
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)

    with pytest.raises(SubmissionError, match="Invalid SkyPilot resources") as exc_info:
        SkyPilotExecutor(job_recovery="RESTART")._validate_submission(
            work_graph=graph,
            pending_work_units=[work_unit],
            workspace=cast("Workspace", _remote_workspace()),
        )

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_multinode_work_unit_sets_num_nodes_and_runs_worker_on_rank_zero(monkeypatch, tmp_path) -> None:
    work_unit = _work_unit(Task(_multinode_task))
    workspace = _remote_workspace()
    fake_sky = _fake_sky(launch_results=(([88], None),))
    snapshot = SimpleNamespace(
        submission_id="MULTINODEABC",
        prepare_job=MagicMock(return_value=("job-1", ["python", "worker.py"], {}, tmp_path / "multi.log")),
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)

    job = SkyPilotExecutor()._dispatch(
        work_unit=work_unit,
        dependencies=set(),
        workspace=cast("Workspace", workspace),
        snapshot=cast("Any", snapshot),
    )

    sky_task = cast("_FakeTask", fake_sky.jobs.launch.call_args.args[0])
    assert sky_task.num_nodes == 2
    assert '"${SKYPILOT_NODE_RANK:-0}" != "0"' in cast("str", sky_task.run)
    assert job.managed_job_id is None
    assert job.request_id == "launch-request-0"


def test_dask_client_work_unit_passes_submission_preflight(monkeypatch) -> None:
    work_unit = _work_unit(Task(_dask_task, DASK_CLIENT))
    graph: DependencyGraph[WorkUnit] = DependencyGraph()
    graph.add_node(work_unit)
    sdk_attempted = False

    def load_sdk() -> object:
        nonlocal sdk_attempted
        sdk_attempted = True
        return _fake_sky()

    monkeypatch.setattr(skypilot_module, "_load_skypilot", load_sdk)

    SkyPilotExecutor()._validate_submission(
        work_graph=graph,
        pending_work_units=[work_unit],
        workspace=cast("Workspace", _remote_workspace()),
    )

    assert sdk_attempted


def test_dask_multinode_work_unit_starts_managed_cluster_on_all_nodes(monkeypatch, tmp_path) -> None:
    task = Task(_dask_task, DASK_CLIENT).with_resources(nodes=3, cpus=4, memory=12)
    work_unit = _work_unit(task)
    workspace = _remote_workspace()
    fake_sky = _fake_sky(launch_results=(([89], None),))
    snapshot = SimpleNamespace(
        submission_id="DASKMULTIABC",
        prepare_job=MagicMock(
            return_value=(
                "job-dask",
                ["python", "worker.py", "--value", "spaces and 'quotes'"],
                {"MISEN_TEST_VALUE": "value with spaces"},
                tmp_path / "dask.log",
            )
        ),
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)

    job = SkyPilotExecutor(dask_startup_timeout=45, dask_scheduler_port=18786)._dispatch(
        work_unit=work_unit,
        dependencies=set(),
        workspace=cast("Workspace", workspace),
        snapshot=cast("Any", snapshot),
    )

    sky_task = cast("_FakeTask", fake_sky.jobs.launch.call_args.args[0])
    script = cast("str", sky_task.run)
    assert sky_task.num_nodes == 3
    assert "SKYPILOT_NODE_RANK" in script
    assert "SKYPILOT_NODE_IPS" in script
    assert "MISEN_DASK_ROLE=scheduler" in script
    assert "MISEN_DASK_ROLE=worker" in script
    assert "MISEN_DASK_ROLE=coordinator" not in script
    assert "MISEN_DASK_EXPECTED_WORKERS=3" in script
    assert "MISEN_DASK_STARTUP_TIMEOUT=45" in script
    assert "MISEN_DASK_CPUS=4" in script
    assert "MISEN_DASK_MEMORY_GIB=12" in script
    assert "18786" in script
    assert 'if [[ "${SKYPILOT_NODE_RANK:-0}" != "0" ]]; then exit 0; fi' not in script
    assert "trap cleanup EXIT" in script
    assert "kill $coordinator_pid $preflight_pid $worker_pid $scheduler_pid" in script
    assert job.managed_job_id is None
    assert job.request_id == "launch-request-0"


@pytest.mark.parametrize("timeout", [True, 0, -1, 1.5])
def test_skypilot_validates_dask_startup_timeout_eagerly(timeout: Any) -> None:
    with pytest.raises(ValueError, match="dask_startup_timeout must be a positive integer"):
        SkyPilotExecutor(dask_startup_timeout=timeout)


@pytest.mark.parametrize("port", [True, 1023, 65536, 1.5])
def test_skypilot_validates_dask_scheduler_port_eagerly(port: Any) -> None:
    with pytest.raises(ValueError, match="dask_scheduler_port must be an integer between 1024 and 65535"):
        SkyPilotExecutor(dask_scheduler_port=port)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"snapshot": False}, "requires snapshot=True"),
        ({"prewarm_envs": True}, "requires prewarm_envs=False"),
    ],
)
def test_remote_only_snapshot_modes_are_rejected_at_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        SkyPilotExecutor(**kwargs)


@pytest.mark.parametrize(
    ("transport", "temp_dir", "message"),
    [
        (None, Path(".cache/misen"), "remotely fetchable workspace transport"),
        ("fetch", Path("/submitter-only/cache"), "relative workspace cache_dir"),
    ],
)
def test_local_only_workspace_shapes_fail_before_sdk_load(transport, temp_dir, message, monkeypatch) -> None:
    work_unit = _work_unit(Task(_cpu_task))
    graph: DependencyGraph[WorkUnit] = DependencyGraph()
    graph.add_node(work_unit)
    workspace = _remote_workspace()
    workspace.bootstrap_transport.return_value = transport
    workspace.get_temp_dir.return_value = temp_dir
    sdk_attempted = False

    def unexpected_sdk() -> object:
        nonlocal sdk_attempted
        sdk_attempted = True
        return object()

    monkeypatch.setattr(skypilot_module, "_load_skypilot", unexpected_sdk)

    with pytest.raises(ConfigError, match=message):
        SkyPilotExecutor()._validate_submission(
            work_graph=graph,
            pending_work_units=[work_unit],
            workspace=cast("Workspace", workspace),
        )

    assert not sdk_attempted


def test_workspace_without_coordination_reads_fails_before_sdk_load(monkeypatch) -> None:
    work_unit = _work_unit(Task(_cpu_task))
    graph: DependencyGraph[WorkUnit] = DependencyGraph()
    graph.add_node(work_unit)
    workspace = _remote_workspace()
    workspace.supports_job_file_reads.return_value = False
    sdk_attempted = False

    def unexpected_sdk() -> object:
        nonlocal sdk_attempted
        sdk_attempted = True
        return object()

    monkeypatch.setattr(skypilot_module, "_load_skypilot", unexpected_sdk)

    with pytest.raises(ConfigError, match="submission-file coordination reads"):
        SkyPilotExecutor()._validate_submission(
            work_graph=graph,
            pending_work_units=[work_unit],
            workspace=cast("Workspace", workspace),
        )

    assert not sdk_attempted


class _FakeManagedStatus(Enum):
    SUCCEEDED = "SUCCEEDED"


def test_bulk_status_batches_active_launch_requests_without_waiting(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    fake_sky = _fake_sky(
        launch_results=(([42], None), ([43], None)),
        request_statuses=("PENDING", "RUNNING"),
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    jobs = [
        SkyPilotJob(
            work_unit=_work_unit(Task(_chain_task, value=index)),
            job_id=f"local-{index}",
            managed_job_id=None,
            request_id=f"launch-request-{index - 1}",
            submission_id="STATUSABC",
            deadline_minutes=60,
            log_path=tmp_path / f"{index}.log",
            workspace=cast("Workspace", workspace),
        )
        for index in (1, 2)
    ]

    assert SkyPilotJob.bulk_state(jobs) == dict.fromkeys(jobs, "pending")
    fake_sky.api_status.assert_called_once_with(request_ids=["launch-request-0", "launch-request-1"])
    fake_sky.get.assert_not_called()
    fake_sky.jobs.queue_v2.assert_not_called()


def test_bulk_status_resolves_launch_request_and_refreshes_durable_record(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    workspace.read_job_file.side_effect = None
    workspace.read_job_file.return_value = msgspec.json.encode(
        executor_module._JobRecord(
            "local-1",
            "launch-request-0",
            "STATUSABC",
            60,
            request_id="launch-request-0",
            native_name="misen-statusabc-local-1",
        )
    )
    fake_sky = _fake_sky(
        launch_results=(([42], None),),
        queue_records=({"job_id": 42, "task_id": 0, "status": "RUNNING"},),
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="launch-request-0",
        managed_job_name="misen-statusabc-local-1",
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "running.log",
        workspace=cast("Workspace", workspace),
    )
    job._bind_record(cast("Workspace", workspace), "record-key")

    assert SkyPilotJob.bulk_state([job]) == {job: "running"}
    assert job.managed_job_id == 42
    assert fake_sky.get.call_args_list == [call("launch-request-0"), call("queue-request")]
    record_call = workspace.put_job_file.call_args_list[0]
    assert record_call.args[:2] == ("jobs", "record-key.json")
    record = msgspec.json.decode(record_call.args[2])
    assert record["native_id"] == 42
    assert record["request_id"] == "launch-request-0"
    assert record["native_name"] == "misen-statusabc-local-1"


def test_bulk_status_recovers_expired_request_by_exact_launch_name(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    managed_name = "misen-statusabc-local-1"
    fake_sky = _fake_sky(
        launch_results=(([42], None),),
        queue_records=({"job_id": 42, "task_id": 0, "job_name": managed_name, "status": "RUNNING"},),
    )
    fake_sky.api_status.return_value = []
    fake_sky.api_status.side_effect = None
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="expired-request",
        managed_job_name=managed_name,
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "running.log",
        workspace=cast("Workspace", workspace),
    )

    assert SkyPilotJob.bulk_state([job]) == {job: "running"}
    assert job.managed_job_id == 42
    assert fake_sky.jobs.queue_v2.call_args_list == [
        call(refresh=True, fields=("job_id", "task_id", "job_name")),
        call(refresh=True, job_ids=[42], fields=("job_id", "task_id", "status", "failure_reason", "end_at")),
    ]


def test_exact_name_recovery_rejects_ambiguous_managed_jobs(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    managed_name = "misen-statusabc-local-1"
    fake_sky = _fake_sky(
        queue_records=(
            {"job_id": 41, "task_id": 0, "job_name": managed_name},
            {"job_id": 42, "task_id": 0, "job_name": managed_name},
        ),
    )
    fake_sky.api_status.return_value = []
    fake_sky.api_status.side_effect = None
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="expired-request",
        managed_job_name=managed_name,
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "unknown.log",
        workspace=cast("Workspace", workspace),
    )

    with pytest.raises(StatusQueryError, match="Multiple SkyPilot managed jobs") as exc_info:
        SkyPilotJob.bulk_state([job])

    assert exc_info.value.retryable is False
    assert job.managed_job_id is None
    workspace.put_job_file.assert_not_called()


@pytest.mark.parametrize(("request_status", "expected_cancellations"), [("FAILED", 0), ("CANCELLED", 1)])
def test_terminal_launch_request_recovers_managed_job_by_exact_name(
    request_status: str,
    expected_cancellations: int,
    tmp_path,
    monkeypatch,
) -> None:
    workspace = _remote_workspace()
    managed_name = "misen-statusabc-local-1"
    fake_sky = _fake_sky(
        launch_results=(RuntimeError("request exited after creating the job"),),
        request_statuses=(request_status,),
        queue_records=({"job_id": 42, "task_id": 0, "job_name": managed_name, "status": "RUNNING"},),
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="launch-request-0",
        managed_job_name=managed_name,
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "running.log",
        workspace=cast("Workspace", workspace),
    )

    assert SkyPilotJob.bulk_state([job]) == {job: "running"}
    assert job.managed_job_id == 42
    assert fake_sky.jobs.cancel.call_count == expected_cancellations
    if expected_cancellations:
        fake_sky.jobs.cancel.assert_called_once_with(job_ids=[42])


def test_cancelled_launch_request_retries_transient_managed_job_cancel_failure(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    managed_name = "misen-statusabc-local-1"
    fake_sky = _fake_sky(
        request_statuses=("CANCELLED",),
        queue_records=({"job_id": 42, "task_id": 0, "job_name": managed_name, "status": "RUNNING"},),
    )
    original_get = fake_sky.get.side_effect
    cancel_attempts = 0
    error_message = "transient cancellation failure"

    def get_with_first_cancel_failure(request_id: str) -> object:
        nonlocal cancel_attempts
        if request_id == "cancel-request":
            cancel_attempts += 1
            if cancel_attempts == 1:
                raise RuntimeError(error_message)
        return original_get(request_id)

    fake_sky.get.side_effect = get_with_first_cancel_failure
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="launch-request-0",
        managed_job_name=managed_name,
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "running.log",
        workspace=cast("Workspace", workspace),
    )

    with pytest.raises(StatusQueryError, match=error_message):
        SkyPilotJob.bulk_state([job])
    assert job.managed_job_id is None

    assert SkyPilotJob.bulk_state([job]) == {job: "running"}
    assert job.managed_job_id == 42
    assert fake_sky.jobs.cancel.call_count == 2


def test_skypilot_job_restores_legacy_and_deferred_durable_records() -> None:
    workspace = cast("Workspace", _remote_workspace())
    work_unit = _work_unit(Task(_chain_task, value=1))

    legacy = SkyPilotJob._from_record(
        work_unit,
        workspace,
        executor_module._JobRecord("local-1", 42, "STATUSABC", 60),
    )
    numeric_string_legacy = SkyPilotJob._from_record(
        work_unit,
        workspace,
        executor_module._JobRecord("local-numeric", "43", "STATUSABC", 60),
    )
    deferred = SkyPilotJob._from_record(
        work_unit,
        workspace,
        executor_module._JobRecord(
            "local-2",
            "launch-request-0",
            "STATUSABC",
            60,
            request_id="launch-request-0",
            native_name="misen-statusabc-local-2",
        ),
    )

    assert legacy.managed_job_id == 42
    assert legacy.request_id is None
    assert numeric_string_legacy.managed_job_id == 43
    assert numeric_string_legacy.request_id is None
    assert deferred.managed_job_id is None
    assert deferred.request_id == "launch-request-0"
    assert deferred.managed_job_name == "misen-statusabc-local-2"


def test_failed_launch_request_becomes_terminal_job_failure(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    fake_sky = _fake_sky(
        launch_results=(RuntimeError("controller rejected request"),),
        request_statuses=("FAILED",),
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="launch-request-0",
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "failed.log",
        workspace=cast("Workspace", workspace),
    )

    assert SkyPilotJob.bulk_state([job]) == {job: "failed"}
    assert "controller rejected request" in cast("str", job.failure.reason)
    workspace.put_job_file.assert_called_once_with("STATUSABC", "local-1.state", b"failed")
    workspace.finalize_job_log.assert_called_once_with(tmp_path / "failed.log")
    fake_sky.jobs.queue_v2.assert_not_called()


def test_failed_launch_request_preserves_committed_success_marker(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    workspace.read_job_file.side_effect = None
    workspace.read_job_file.return_value = b"done"
    fake_sky = _fake_sky(
        launch_results=(RuntimeError("request process exited after launch"),),
        request_statuses=("FAILED",),
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="launch-request-0",
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "done.log",
        workspace=cast("Workspace", workspace),
    )

    assert SkyPilotJob.bulk_state([job]) == {job: "done"}
    assert job.failure.reason is None
    workspace.put_job_file.assert_not_called()
    workspace.finalize_job_log.assert_called_once_with(tmp_path / "done.log")


def test_failed_launch_request_preserves_committed_result_without_marker(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    fake_sky = _fake_sky(
        launch_results=(RuntimeError("request process exited after launch"),),
        request_statuses=("FAILED",),
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)

    def completed(self: WorkUnit, workspace: Workspace) -> bool:
        del self, workspace
        return True

    monkeypatch.setattr(WorkUnit, "done", completed)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="launch-request-0",
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "done.log",
        workspace=cast("Workspace", workspace),
    )

    assert SkyPilotJob.bulk_state([job]) == {job: "done"}
    assert job.failure.reason is None
    workspace.put_job_file.assert_called_once_with("STATUSABC", "local-1.state", b"done")
    workspace.finalize_job_log.assert_called_once_with(tmp_path / "done.log")


def test_failed_launch_request_rechecks_success_before_publishing_failure(tmp_path, monkeypatch) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    failure = RuntimeError("request process exited after launch")
    fake_sky = _fake_sky(
        launch_results=(failure,),
        request_statuses=("FAILED",),
    )
    original_get = fake_sky.get.side_effect

    def get_after_worker_commit(request_id: str) -> object:
        if request_id == "launch-request-0":
            workspace.put_job_file("STATUSABC", "local-1.state", b"done")
        return original_get(request_id)

    fake_sky.get.side_effect = get_after_worker_commit
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="launch-request-0",
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "done.log",
        workspace=workspace,
    )

    assert SkyPilotJob.bulk_state([job]) == {job: "done"}
    assert workspace.read_job_file("STATUSABC", "local-1.state") == b"done"
    assert job.failure.reason is None


def test_expired_failed_launch_request_recovers_workspace_marker(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    workspace.read_job_file.side_effect = None
    workspace.read_job_file.return_value = b"failed"
    fake_sky = _fake_sky(queue_records=())
    fake_sky.api_status.return_value = []
    fake_sky.api_status.side_effect = None
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="expired-request",
        managed_job_name="misen-statusabc-local-1",
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "failed.log",
        workspace=cast("Workspace", workspace),
    )

    assert SkyPilotJob.bulk_state([job]) == {job: "failed"}
    assert "workspace recorded the job as failed" in cast("str", job.failure.reason)
    workspace.finalize_job_log.assert_called_once_with(tmp_path / "failed.log")
    fake_sky.jobs.queue_v2.assert_not_called()


def test_malformed_launch_result_is_immediately_nonretryable(tmp_path, monkeypatch) -> None:
    fake_sky = _fake_sky(launch_results=({"unexpected": "shape"},))
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="launch-request-0",
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "failed.log",
        workspace=cast("Workspace", _remote_workspace()),
    )

    with pytest.raises(StatusQueryError, match="unexpected result") as exc_info:
        SkyPilotJob.bulk_state([job])

    assert exc_info.value.retryable is False


def test_managed_id_persistence_failure_retries_without_resolving_again(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    workspace.read_job_file.side_effect = None
    workspace.read_job_file.return_value = msgspec.json.encode(
        executor_module._JobRecord(
            "local-1",
            "launch-request-0",
            "STATUSABC",
            60,
            request_id="launch-request-0",
        )
    )
    workspace.put_job_file.side_effect = [StorageError("object store unavailable"), None]
    fake_sky = _fake_sky(
        launch_results=(([42], None),),
        queue_records=({"job_id": 42, "task_id": 0, "status": "RUNNING"},),
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=None,
        request_id="launch-request-0",
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "running.log",
        workspace=cast("Workspace", workspace),
    )
    job._bind_record(cast("Workspace", workspace), "record-key")

    with pytest.raises(StatusQueryError, match="Could not persist managed-job ID"):
        SkyPilotJob.bulk_state([job])

    assert job.managed_job_id == 42
    fake_sky.jobs.queue_v2.assert_not_called()

    assert SkyPilotJob.bulk_state([job]) == {job: "running"}
    assert job.managed_job_id == 42
    assert fake_sky.get.call_args_list.count(call("launch-request-0")) == 1
    assert workspace.put_job_file.call_count == 2


def test_reattach_persists_resolved_id_when_later_status_query_fails(tmp_path, monkeypatch) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "workspace"))
    fake_sky = _fake_sky(launch_results=(([42], None),))
    snapshot = SimpleNamespace(
        submission_id="STATUSABC",
        snapshot_key="SNAPSHOT",
        prepare_job=MagicMock(return_value=("local-1", ["python", "worker.py"], {}, tmp_path / "job.log")),
    )
    work_unit = _work_unit(Task(_chain_task, value=1))
    executor = SkyPilotExecutor()
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)

    first = executor._dispatch_or_reattach(work_unit, set(), workspace, cast("Any", snapshot))
    record_key = cast("str", first._record_key)
    original_get = fake_sky.get.side_effect
    error_message = "jobs controller unavailable"

    def get_with_queue_failure(request_id: str) -> object:
        if request_id == "queue-request":
            raise RuntimeError(error_message)
        return original_get(request_id)

    fake_sky.get.side_effect = get_with_queue_failure

    with pytest.raises(StatusQueryError, match="Could not query SkyPilot managed jobs"):
        executor._dispatch_or_reattach(work_unit, set(), workspace, cast("Any", snapshot))

    record = msgspec.json.decode(
        workspace.read_job_file("jobs", f"{record_key}.json"),
        type=executor_module._JobRecord,
    )
    assert record.native_id == 42
    assert record.request_id == "launch-request-0"
    fake_sky.jobs.launch.assert_called_once()


def test_stale_job_cannot_overwrite_newer_durable_record(tmp_path) -> None:
    workspace = _remote_workspace()
    workspace.read_job_file.side_effect = None
    workspace.read_job_file.return_value = msgspec.json.encode(
        executor_module._JobRecord(
            "new-job",
            "new-launch-request",
            "NEW",
            60,
            request_id="new-launch-request",
        )
    )
    stale = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="old-job",
        managed_job_id=None,
        request_id="old-launch-request",
        submission_id="OLD",
        deadline_minutes=60,
        log_path=tmp_path / "old.log",
        workspace=cast("Workspace", workspace),
    )
    stale._bind_record(cast("Workspace", workspace), "record-key")

    assert stale._remember_managed_job_id(42) == 42

    workspace.lock.assert_called_once_with("job", "record-key")
    workspace.put_job_file.assert_not_called()


def test_bulk_status_uses_one_request_caches_terminal_states_and_finalizes_once(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    records = [
        {"job_id": 42, "task_id": 0, "status": _FakeManagedStatus.SUCCEEDED},
        {"job_id": 43, "task_id": 0, "status": "FAILED_SETUP", "failure_reason": "setup failed"},
        {"job_id": 7, "task_id": 0, "status": "RECOVERING"},
    ]
    fake_sky = _fake_sky(queue_records=records)
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)

    def incomplete(self: WorkUnit, workspace: Workspace) -> bool:
        del self, workspace
        return False

    monkeypatch.setattr(WorkUnit, "done", incomplete)
    jobs = [
        SkyPilotJob(
            work_unit=_work_unit(Task(_chain_task, value=1)),
            job_id="local-1",
            managed_job_id=42,
            submission_id="STATUSABC",
            deadline_minutes=60,
            log_path=tmp_path / "done.log",
            workspace=cast("Workspace", workspace),
        ),
        SkyPilotJob(
            work_unit=_work_unit(Task(_chain_task, value=2)),
            job_id="local-2",
            managed_job_id=43,
            submission_id="STATUSABC",
            deadline_minutes=60,
            log_path=tmp_path / "failed.log",
            workspace=cast("Workspace", workspace),
        ),
        SkyPilotJob(
            work_unit=_work_unit(Task(_chain_task, value=3)),
            job_id="local-3",
            managed_job_id=7,
            submission_id="STATUSABC",
            deadline_minutes=60,
            log_path=tmp_path / "running.log",
            workspace=cast("Workspace", workspace),
        ),
    ]

    states = SkyPilotJob.bulk_state(jobs)
    repeated_states = SkyPilotJob.bulk_state(jobs)

    assert states == repeated_states == {jobs[0]: "done", jobs[1]: "failed", jobs[2]: "running"}
    assert fake_sky.jobs.queue_v2.call_args_list == [
        call(
            refresh=True,
            job_ids=[7, 42, 43],
            fields=("job_id", "task_id", "status", "failure_reason", "end_at"),
        ),
        call(refresh=True, job_ids=[7], fields=("job_id", "task_id", "status", "failure_reason", "end_at")),
    ]
    assert workspace.put_job_file.call_args_list == [
        call("STATUSABC", "local-1.state", b"done"),
        call("STATUSABC", "local-2.state", b"failed"),
    ]
    assert workspace.finalize_job_log.call_args_list == [
        call(tmp_path / "done.log"),
        call(tmp_path / "failed.log"),
    ]
    assert jobs[0].failure.reason is None
    assert "FAILED_SETUP: setup failed" in cast("str", jobs[1].failure.reason)


@pytest.mark.parametrize("marker", [b"done", b"failed"])
def test_missing_managed_job_history_recovers_workspace_terminal_state(marker: bytes, tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    workspace.read_job_file.side_effect = None
    workspace.read_job_file.return_value = marker
    fake_sky = _fake_sky(queue_records=())
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=42,
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "terminal.log",
        workspace=cast("Workspace", workspace),
    )

    assert SkyPilotJob.bulk_state([job]) == {job: marker.decode()}
    workspace.put_job_file.assert_not_called()
    workspace.finalize_job_log.assert_called_once_with(tmp_path / "terminal.log")
    if marker == b"failed":
        assert "workspace recorded the job as failed" in cast("str", job.failure.reason)


def test_controller_failure_waits_for_authoritative_worker_state(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    fake_sky = _fake_sky(
        queue_records=({"job_id": 42, "task_id": 0, "status": "FAILED_CONTROLLER", "end_at": time.time()},),
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)

    def incomplete(self: WorkUnit, workspace: Workspace) -> bool:
        del self, workspace
        return False

    monkeypatch.setattr(WorkUnit, "done", incomplete)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=42,
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "done.log",
        workspace=cast("Workspace", workspace),
    )

    assert SkyPilotJob.bulk_state([job]) == {job: "running"}
    workspace.put_job_file.assert_not_called()
    workspace.finalize_job_log.assert_not_called()

    workspace.read_job_file.side_effect = None
    workspace.read_job_file.return_value = b"done"
    assert SkyPilotJob.bulk_state([job]) == {job: "done"}
    workspace.finalize_job_log.assert_called_once_with(tmp_path / "done.log")


def test_controller_failure_publishes_terminal_gate_after_worker_deadline(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    end_at = time.time() - 60 * 60 - skypilot_module._CONTROLLER_FAILURE_KILL_GRACE_S - 1
    fake_sky = _fake_sky(
        queue_records=({"job_id": 42, "task_id": 0, "status": "FAILED_CONTROLLER", "end_at": end_at},),
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)

    def incomplete(self: WorkUnit, workspace: Workspace) -> bool:
        del self, workspace
        return False

    monkeypatch.setattr(WorkUnit, "done", incomplete)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=42,
        request_id="launch-request-0",
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "failed.log",
        workspace=cast("Workspace", workspace),
    )

    assert SkyPilotJob.bulk_state([job]) == {job: "failed"}
    workspace.put_job_file.assert_called_once_with("STATUSABC", "local-1.state", b"failed")


def test_legacy_controller_failure_converges_after_worker_deadline(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    record = {"job_id": 42, "task_id": 0, "status": "FAILED_CONTROLLER", "end_at": time.time()}
    fake_sky = _fake_sky(
        queue_records=(record,),
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)

    def incomplete(self: WorkUnit, workspace: Workspace) -> bool:
        del self, workspace
        return False

    monkeypatch.setattr(WorkUnit, "done", incomplete)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=42,
        submission_id="STATUSABC",
        deadline_minutes=0,
        log_path=tmp_path / "failed.log",
        workspace=cast("Workspace", workspace),
    )

    assert SkyPilotJob.bulk_state([job]) == {job: "running"}
    record["end_at"] = time.time() - skypilot_module._CONTROLLER_FAILURE_KILL_GRACE_S - 1

    restored_job = SkyPilotJob(
        work_unit=job.work_unit,
        job_id="local-1",
        managed_job_id=42,
        submission_id="STATUSABC",
        deadline_minutes=0,
        log_path=tmp_path / "failed.log",
        workspace=cast("Workspace", workspace),
    )

    assert SkyPilotJob.bulk_state([restored_job]) == {restored_job: "failed"}
    workspace.put_job_file.assert_called_once_with("STATUSABC", "local-1.state", b"failed")


def test_controller_failure_defers_to_committed_workspace_result(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    fake_sky = _fake_sky(
        queue_records=[{"job_id": 42, "task_id": 0, "status": "FAILED_CONTROLLER", "failure_reason": "controller lost"}]
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)

    def completed(self: WorkUnit, workspace: Workspace) -> bool:
        del self, workspace
        return True

    monkeypatch.setattr(WorkUnit, "done", completed)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=42,
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "done.log",
        workspace=cast("Workspace", workspace),
    )

    assert SkyPilotJob.bulk_state([job]) == {job: "done"}
    assert job.failure.reason is None
    assert workspace.finalize_job_log.call_args_list == [call(tmp_path / "done.log")]


def test_controller_failure_defers_to_committed_dependency_marker(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    workspace.read_job_file.side_effect = None
    workspace.read_job_file.return_value = b"done"
    fake_sky = _fake_sky(
        queue_records=[{"job_id": 42, "task_id": 0, "status": "FAILED_CONTROLLER", "failure_reason": "lost"}]
    )
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)

    def unexpected_cache_check(self: WorkUnit, workspace: Workspace) -> bool:
        del self, workspace
        msg = "the terminal marker is already authoritative"
        raise AssertionError(msg)

    monkeypatch.setattr(WorkUnit, "done", unexpected_cache_check)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=42,
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "done.log",
        workspace=cast("Workspace", workspace),
    )

    assert SkyPilotJob.bulk_state([job]) == {job: "done"}
    workspace.put_job_file.assert_not_called()


def test_controller_failure_retries_an_uncertain_dependency_marker_read(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    workspace.read_job_file.side_effect = StorageError("object store unavailable")
    fake_sky = _fake_sky(queue_records=[{"job_id": 42, "task_id": 0, "status": "FAILED_CONTROLLER"}])
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=42,
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=tmp_path / "failed.log",
        workspace=cast("Workspace", workspace),
    )

    with pytest.raises(StatusQueryError, match="completion marker") as exc_info:
        SkyPilotJob.bulk_state([job])

    assert exc_info.value.retryable
    workspace.put_job_file.assert_not_called()


def test_malformed_queue_response_is_immediately_nonretryable(monkeypatch) -> None:
    fake_sky = _fake_sky()
    fake_sky.get = MagicMock(return_value={"unexpected": "shape"})
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    workspace = _remote_workspace()
    job = SkyPilotJob(
        work_unit=_work_unit(Task(_chain_task, value=1)),
        job_id="local-1",
        managed_job_id=42,
        submission_id="STATUSABC",
        deadline_minutes=60,
        log_path=Path(".cache/misen/job.log"),
        workspace=cast("Workspace", workspace),
    )

    with pytest.raises(StatusQueryError, match="unexpected response") as exc_info:
        SkyPilotJob.bulk_state([job])

    assert exc_info.value.retryable is False


def test_installed_skypilot_sdk_contract() -> None:
    if importlib.util.find_spec("sky") is None:
        pytest.skip("SkyPilot is not installed")

    probe = textwrap.dedent(
        """
        import sky
        import inspect

        resource = sky.Resources(
            infra="aws/us-east-1",
            cpus="2+",
            memory="4+",
            use_spot=True,
            max_hourly_cost=1.0,
            job_recovery="FAILOVER",
        )
        resource.validate()
        task = sky.Task(
            name="misen-contract-probe",
            run="true",
            num_nodes=2,
            resources=[resource],
            api_server_access=False,
        )

        assert task.num_nodes == 2
        assert task.api_server_access is False
        assert len(task.resources) == 1
        assert callable(sky.jobs.launch)
        inspect.signature(sky.jobs.launch).bind(task, name="misen-contract-probe", pool="misen-pool")
        assert callable(sky.jobs.queue_v2)
        assert callable(sky.get)
        assert callable(sky.api_status)
        inspect.signature(sky.api_cancel).bind(request_ids=["request-id"], silent=True)
        assert callable(sky.server.common.is_api_server_local)
        assert callable(sky.server.common.check_server_healthy_or_start_fn)
        assert callable(sky.server.common.check_server_healthy)
        assert callable(sky.server.common.get_server_url)
        assert callable(sky.skypilot_config.get_nested)
        from sky.skylet import constants
        assert isinstance(constants.API_SERVER_CREATION_LOCK_PATH, str)
        assert isinstance(constants.ENV_VAR_IS_SKYPILOT_SERVER, str)
        """
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_missing_skypilot_sdk_has_actionable_lazy_import_error(monkeypatch) -> None:
    def missing_sky(module_name: str) -> object:
        assert module_name == "sky"
        msg = "No module named 'sky'"
        raise ModuleNotFoundError(msg, name="sky")

    monkeypatch.setattr(skypilot_module.importlib, "import_module", missing_sky)

    with pytest.raises(ConfigError, match=r"SkyPilotExecutor requires SkyPilot >=0\.13") as exc_info:
        skypilot_module._load_skypilot()

    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)
    assert "misen[skypilot]" in str(exc_info.value)


def test_skypilot_transitive_import_error_is_not_hidden(monkeypatch) -> None:
    error = ModuleNotFoundError("No module named 'sky_dependency'", name="sky_dependency")

    def broken_sky_import(_module_name: str) -> object:
        raise error

    monkeypatch.setattr(skypilot_module.importlib, "import_module", broken_sky_import)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        skypilot_module._load_skypilot()

    assert exc_info.value is error


def test_skypilot_sdk_version_is_not_rejected_at_runtime(monkeypatch) -> None:
    future_sdk = SimpleNamespace(__version__="99.0.0")
    monkeypatch.setattr(skypilot_module.importlib, "import_module", lambda _name: future_sdk)

    assert skypilot_module._load_skypilot() is future_sdk
