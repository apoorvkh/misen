"""SkyPilot executor tests using a hermetic in-process SDK fake."""
# ruff: noqa: ANN001, D103, PLR2004, S101, SLF001

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, call

import pytest

import misen.executor as executor_module
import misen.executors.skypilot as skypilot_module
import misen.utils.snapshot as snapshot_module
from misen import DASK_CLIENT, Task, meta
from misen.exceptions import ConfigError, StatusQueryError, StorageError, SubmissionError
from misen.executors.skypilot import SkyPilotExecutor, SkyPilotJob
from misen.utils.graph import DependencyGraph
from misen.utils.work_unit import WorkUnit
from misen.workspace import Workspace
from misen.workspaces.memory import InMemoryWorkspace

if TYPE_CHECKING:
    from collections.abc import Sequence

    from misen.executor import CompletedJob


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


def _fake_sky(
    *,
    launch_results: Sequence[object] = (([42], None),),
    queue_records: Sequence[object] = (),
    api_server_local: bool = True,
) -> SimpleNamespace:
    requests: dict[str, object] = {f"launch-request-{index}": result for index, result in enumerate(launch_results)}
    launch_count = 0

    def launch(_task: object, *, name: str) -> str:
        nonlocal launch_count
        del name
        request_id = f"launch-request-{launch_count}"
        launch_count += 1
        return request_id

    jobs = SimpleNamespace(
        cancel=MagicMock(return_value="cancel-request"),
        launch=MagicMock(side_effect=launch),
        queue_v2=MagicMock(return_value="queue-request"),
    )

    def get(request_id: str) -> object:
        if request_id == "cancel-request":
            return None
        if request_id == "queue-request":
            return (list(queue_records), "ignored-metadata")
        if request_id in requests:
            return requests[request_id]
        msg = f"Unexpected fake SkyPilot request: {request_id}"
        raise AssertionError(msg)

    return SimpleNamespace(
        Task=_FakeTask,
        Resources=_FakeResources,
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
    assert [(job.managed_job_id, job.job_id) for job in jobs] == [
        (311, "misen-job-1"),
        (312, "misen-job-2"),
        (313, "misen-job-3"),
        (314, "misen-job-4"),
    ]
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

    def fail_second_launch(task: object, *, name: str) -> str:
        del task
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
    assert submitted.managed_job_id == 101


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
    assert job.managed_job_id == 88


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
    assert job.managed_job_id == 89


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


def test_bulk_status_uses_one_request_caches_terminal_states_and_finalizes_once(tmp_path, monkeypatch) -> None:
    workspace = _remote_workspace()
    records = [
        {"job_id": 42, "task_id": 0, "status": _FakeManagedStatus.SUCCEEDED},
        {"job_id": 43, "task_id": 0, "status": "FAILED_CONTROLLER", "failure_reason": "controller lost"},
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
        call(refresh=True, job_ids=[7, 42, 43], fields=("job_id", "task_id", "status", "failure_reason")),
        call(refresh=True, job_ids=[7], fields=("job_id", "task_id", "status", "failure_reason")),
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
    assert "FAILED_CONTROLLER: controller lost" in cast("str", jobs[1].failure.reason)


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
    assert workspace.put_job_file.call_args_list == [call("STATUSABC", "local-1.state", b"done")]


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
        assert callable(sky.jobs.queue_v2)
        assert callable(sky.get)
        assert callable(sky.server.common.is_api_server_local)
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

    with pytest.raises(ConfigError, match=r"SkyPilotExecutor requires SkyPilot >=0\.12\.1") as exc_info:
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
