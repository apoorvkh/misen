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
from typing import TYPE_CHECKING, Any, ClassVar, cast
from unittest.mock import MagicMock, call

import msgspec
import pytest

import misen.executor as executor_module
import misen.executors.skypilot as graph_module
import misen.executors.skypilot as skypilot_module
import misen.utils.snapshot as snapshot_module
from misen import DASK_CLIENT, Task, meta
from misen.exceptions import CacheError, ConfigError, ExecutionError, StatusQueryError, StorageError
from misen.executors.skypilot import RunManifest, SkyPilotCapacity, SkyPilotExecutor, SkyPilotJob, SkyPilotTaskJob
from misen.utils.graph import DependencyGraph
from misen.utils.work_unit import WorkUnit
from misen.workspace import Workspace
from misen.workspaces.memory import InMemoryWorkspace

if TYPE_CHECKING:
    from collections.abc import Sequence


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

    __slots__ = ()


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


@pytest.mark.parametrize("cached_parent", [False, True])
def test_submit_builds_logical_diamond_and_one_bounded_agent_without_eager_launch(
    monkeypatch, *, cached_parent: bool
) -> None:
    graph, work_units = _diamond_graph()
    parent = work_units[0]
    workspace = _remote_workspace()
    prepared: list[tuple[WorkUnit, object]] = []
    staged_agents: list[object] = []
    fake_sky = _fake_sky()

    class FakeSnapshot:
        submission_id = "GRAPHABC"
        snapshot_key = "GRAPH-SNAPSHOT"

        def __init__(self, **_kwargs: object) -> None:
            pass

        def prepare_job(
            self, work_unit: WorkUnit, workspace: Workspace, *, dependency_jobs: object = None
        ) -> tuple[str, list[str], dict[str, str], Path]:
            del workspace
            prepared.append((work_unit, dependency_jobs))
            task_id = work_unit.root.kwargs["value"]
            return (
                f"logical-{task_id}",
                ["python", "worker.py", str(task_id)],
                {"TASK_VALUE": str(task_id)},
                Path(f"logs/{task_id}.log"),
            )

    def prepare_agent(
        _snapshot: object, _workspace: Workspace, fn: object
    ) -> tuple[str, list[str], dict[str, str], Path]:
        staged_agents.append(fn)
        return "agent-bootstrap", ["python", "agent.py"], {}, Path("logs/agent.log")

    monkeypatch.setattr(executor_module, "build_work_graph", lambda **_kwargs: graph)

    def done(self: WorkUnit, workspace: Workspace) -> bool:
        del workspace
        return cached_parent and self is parent

    monkeypatch.setattr(WorkUnit, "done", done)
    monkeypatch.setattr(snapshot_module, "ProjectSnapshot", FakeSnapshot)
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    monkeypatch.setattr(graph_module, "_prepare_control", prepare_agent)
    monkeypatch.setattr(graph_module, "_SkyCapacityBackend", MagicMock())
    start = MagicMock()
    close = MagicMock()
    monkeypatch.setattr(graph_module.GraphCoordinator, "start", start)
    monkeypatch.setattr(graph_module.GraphCoordinator, "close", close)
    executor = SkyPilotExecutor(capacity={"cpu": {"pool": "misen-dev"}}, manage_api_server=False)

    with executor.session():
        result = executor.submit({Task(_chain_task, value=99)}, cast("Workspace", workspace))

    pending = [unit for unit in work_units if not (cached_parent and unit is parent)]
    assert prepared == [(unit, None) for unit in pending]
    assert len(staged_agents) == 1
    start.assert_called_once()
    close.assert_called_once()
    fake_sky.jobs.launch.assert_not_called()
    fake_sky.get.assert_not_called()
    records = [
        msgspec.json.decode(item.args[2], type=RunManifest)
        for item in workspace.put_job_file.call_args_list
        if item.args[1] == "run-manifest.json"
    ]
    assert len(records) == 1
    manifest = records[0]
    nodes = {node.job_id: node for node in manifest.nodes}
    assert nodes["logical-2"].dependencies == ([] if cached_parent else ["logical-1"])
    assert nodes["logical-3"].dependencies == ([] if cached_parent else ["logical-1"])
    assert set(nodes["logical-4"].dependencies) == {"logical-2", "logical-3"}
    assert all(node.profile == "cpu" for node in nodes.values())
    assert len(manifest.agents) == 1
    assert len(list(result)) == 4
    assert all(isinstance(result[work_units.index(unit)], SkyPilotTaskJob) for unit in pending)
    assert result.successors(1) == [result[0]]
    assert result.successors(2) == [result[0]]
    assert set(result.successors(3)) == {result[1], result[2]}
    if cached_parent:
        assert isinstance(result[0], executor_module.CompletedJob)


def test_public_executor_has_one_graph_contract_and_decodes_capacity_profiles() -> None:
    executor = msgspec.convert(
        {"capacity": {"cpu": {"pool": "misen-cpu", "cpus": 4, "memory": 32, "max_workers": 2}}},
        type=SkyPilotExecutor,
    )

    assert isinstance(executor, graph_module.GraphSkyPilotExecutor)
    assert isinstance(executor.capacity["cpu"], SkyPilotCapacity)
    assert executor.capacity["cpu"].pool == "misen-cpu"
    assert executor.capacity["cpu"].max_workers == 2
    assert executor.lifecycle == "attached"


@pytest.mark.parametrize(
    "kwargs", [{"infra": "aws"}, {"pool": "misen-dev"}, {"mode": "managed"}, {"execution_mode": "cluster"}]
)
def test_obsolete_per_job_options_are_not_silent_execution_modes(kwargs: dict[str, Any]) -> None:
    with pytest.raises(TypeError, match="Unexpected keyword argument"):
        SkyPilotExecutor(**kwargs)


@pytest.mark.parametrize("name", ["", "../other", "with space", "x" * 65])
def test_capacity_profile_names_are_bounded(name: str) -> None:
    with pytest.raises(ValueError, match="Capacity names"):
        SkyPilotExecutor(capacity={name: SkyPilotCapacity(pool="workers")})


def test_resource_routing_prefers_cpu_capacity_and_uses_explicit_accelerator_shapes() -> None:
    executor = SkyPilotExecutor(
        capacity={
            "gpu": SkyPilotCapacity(pool="gpu", cpus=8, memory=64, accelerators={"A100": 2}, accelerator_memory=80),
            "cpu": SkyPilotCapacity(pool="cpu", cpus=4, memory=32),
            "too-small-gpu": SkyPilotCapacity(
                pool="small", cpus=8, memory=64, accelerators={"L4": 2}, accelerator_memory=24
            ),
        },
    )

    assert executor._profile(_work_unit(Task(_cpu_task))) == "cpu"
    assert executor._profile(_work_unit(Task(_gpu_task))) == "gpu"
    with pytest.raises(ConfigError, match="No configured SkyPilot capacity fits"):
        SkyPilotExecutor(capacity={"cpu": executor.capacity["cpu"]})._profile(_work_unit(Task(_gpu_task)))


def test_empty_capacity_does_not_implicitly_provision_cloud_resources() -> None:
    with pytest.raises(ConfigError, match="explicit bounded profile"):
        SkyPilotExecutor()._profile(_work_unit(Task(_cpu_task)))


@pytest.mark.parametrize("backend", ["tpu", "mps"])
def test_unsupported_accelerator_backends_fail_before_allocation(monkeypatch, backend: str) -> None:
    profile = SkyPilotCapacity(
        pool="accelerators", accelerators={"declared-device": 1}, accelerator_type=cast("Any", backend)
    )
    executor = SkyPilotExecutor(capacity={"devices": profile}, manage_api_server=False)
    work_unit = _work_unit(
        Task(_gpu_task).with_resources(
            cpus=1, memory=8, accelerators=1, accelerator_type=cast("Any", backend), accelerator_memory=None
        )
    )
    loader = MagicMock(side_effect=AssertionError("Unsupported device isolation must fail before contacting SkyPilot"))
    monkeypatch.setattr(skypilot_module, "_load_skypilot", loader)

    with executor.session(), pytest.raises(ConfigError, match=r"(accelerator|cuda|rocm|xpu)"):
        executor._validate_submission(
            work_graph=DependencyGraph(),
            pending_work_units=[work_unit],
            workspace=cast("Workspace", _remote_workspace()),
        )
    loader.assert_not_called()


def test_multinode_and_dask_requests_select_dedicated_reserved_topology() -> None:
    executor = SkyPilotExecutor(
        capacity={
            "single": SkyPilotCapacity(pool="single"),
            "two": SkyPilotCapacity(infra="aws", nodes=2, dedicated=True),
            "four": SkyPilotCapacity(infra="aws", nodes=4, dedicated=True),
        },
    )
    assert executor._profile(_work_unit(Task(_multinode_task))) == "two"
    assert executor._profile(_work_unit(Task(_dask_task, DASK_CLIENT))) == "two"
    with pytest.raises(ConfigError, match="No configured SkyPilot capacity fits"):
        SkyPilotExecutor(capacity={"four": executor.capacity["four"]})._profile(
            _work_unit(Task(_dask_task, DASK_CLIENT))
        )


def test_attached_submission_preflight_requires_an_owned_session(monkeypatch) -> None:
    work_unit = _work_unit(Task(_cpu_task))
    executor = SkyPilotExecutor(
        capacity={"cpu": SkyPilotCapacity(pool="cpu", cpus=4, memory=32)}, manage_api_server=False
    )
    load = MagicMock(side_effect=AssertionError("Preflight should not contact SkyPilot"))
    monkeypatch.setattr(skypilot_module, "_load_skypilot", load)
    kwargs = {
        "work_graph": DependencyGraph(),
        "pending_work_units": [work_unit],
        "workspace": cast("Workspace", _remote_workspace()),
    }

    with pytest.raises(ConfigError, match="Nonblocking attached submissions require"):
        executor._validate_submission(**kwargs)
    with executor.session():
        executor._validate_submission(**kwargs)
    load.assert_not_called()


def test_nested_sessions_preserve_owner_and_do_not_start_local_service(monkeypatch) -> None:
    load = MagicMock(side_effect=AssertionError("Empty sessions must remain lazy"))
    monkeypatch.setattr(skypilot_module, "_load_skypilot", load)
    executor = SkyPilotExecutor(manage_api_server=False)

    with executor.session():
        owner = graph_module._runs.get()
        assert owner is not None
        assert owner[0] == id(executor)
        with executor.session():
            assert graph_module._runs.get() is owner
        assert graph_module._runs.get() is owner
    assert graph_module._runs.get() is None
    load.assert_not_called()


def test_blocking_submit_owns_session_around_the_whole_graph(monkeypatch) -> None:
    executor = SkyPilotExecutor(manage_api_server=False)
    expected: DependencyGraph[Any] = DependencyGraph()

    def submit(self, tasks, workspace, *, blocking=False) -> DependencyGraph[Any]:
        del tasks, workspace
        assert self is executor
        assert blocking
        owner = graph_module._runs.get()
        assert owner is not None
        assert owner[0] == id(executor)
        return expected

    monkeypatch.setattr(executor_module.Executor, "submit", submit)
    assert executor.submit(set(), cast("Workspace", _remote_workspace()), blocking=True) is expected
    assert graph_module._runs.get() is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lifecycle": "detached"},
        {"lifecycle": "detached", "manage_api_server": False},
        {"coordinator": SkyPilotCapacity(infra="aws", dedicated=True)},
        {
            "lifecycle": "detached",
            "manage_api_server": False,
            "coordinator": SkyPilotCapacity(pool="borrowed", dedicated=True),
        },
    ],
)
def test_detached_lifecycle_requires_explicit_run_owned_coordinator(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"(Detached runs require|Coordinator capacity)"):
        SkyPilotExecutor(**kwargs)


def test_detached_preflight_rejects_local_api_before_any_launch(monkeypatch) -> None:
    fake_sky = _fake_sky()
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: fake_sky)
    executor = SkyPilotExecutor(
        capacity={"cpu": SkyPilotCapacity(pool="cpu", cpus=4, memory=32)},
        lifecycle="detached",
        manage_api_server=False,
        coordinator=SkyPilotCapacity(infra="aws", dedicated=True),
    )
    with pytest.raises(ConfigError, match="stable remote SkyPilot API"):
        executor._validate_submission(
            work_graph=DependencyGraph(),
            pending_work_units=[_work_unit(Task(_cpu_task))],
            workspace=cast("Workspace", _remote_workspace()),
        )
    fake_sky.jobs.launch.assert_not_called()


@pytest.mark.parametrize("name", ["setup_timeout_s", "shutdown_timeout_s", "poll_interval_s"])
@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan")])
def test_lifecycle_timeouts_are_finite_and_positive(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=name):
        SkyPilotExecutor(**{name: value})


@pytest.mark.parametrize("minutes", [True, 0, -1, 1.5])
def test_run_lifetime_is_explicitly_bounded(minutes: object) -> None:
    with pytest.raises(ValueError, match="max_run_minutes"):
        SkyPilotExecutor(max_run_minutes=cast("Any", minutes))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"snapshot": False}, "requires snapshot=True"),
        ({"prewarm_envs": True}, "prewarm_envs=False"),
    ],
)
def test_remote_only_snapshot_modes_are_rejected_at_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        SkyPilotExecutor(**kwargs)


@pytest.mark.parametrize(
    ("transport", "temp_dir", "message"),
    [
        (None, Path(".cache/misen"), "remotely fetchable workspace"),
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

    with pytest.raises(ConfigError, match="job-file coordination"):
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

    class NativeRecordExecutor(executor_module.Executor[SkyPilotJob]):
        """Exercise generic native-record restoration without graph submission."""

        _job_class: ClassVar[type[executor_module.Job] | None] = SkyPilotJob

        def _dispatch(self, work_unit, dependencies, workspace, snapshot) -> SkyPilotJob:
            del dependencies
            return SkyPilotJob(
                work_unit=work_unit,
                job_id="local-1",
                managed_job_id=None,
                request_id=str(fake_sky.jobs.launch(None, name="native-allocation", pool=None)),
                managed_job_name="native-allocation",
                submission_id=snapshot.submission_id,
                deadline_minutes=60,
                log_path=tmp_path / "job.log",
                workspace=workspace,
            )

    work_unit = _work_unit(Task(_chain_task, value=1))
    executor = NativeRecordExecutor()
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
