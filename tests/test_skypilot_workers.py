"""Reusable SkyPilot workers: scheduling, GPU eligibility, and recovery."""
# ruff: noqa: ANN001, ANN201, D103, PLR2004, S101, SLF001

from __future__ import annotations

import contextlib
import inspect
import subprocess
import tomllib
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cloudpickle
import msgspec
import pytest

import misen.executors.skypilot as skypilot_module
import misen.executors.skypilot as controller_module
from misen import Task
from misen.exceptions import CacheError, ConfigError, ExecutionError
from misen.executor import CompletedJob
from misen.executors.skypilot import SkyPilotExecutor, SkyPilotJob, SkyPilotWorker
from misen.task_metadata import aggregate_resources
from misen.executors.skypilot import Controller, ControllerState, WorkerState, WorkSpec, WorkState
from misen.executors.skypilot import WorkerOffering, resolve_workers, verify_accelerators
from misen.utils.work_unit import WorkUnit
from tests.test_skypilot_executor import _chain_task, _diamond_graph, _remote_workspace


class Store(msgspec.Struct, dict=True):
    def __post_init__(self):
        self.files = {}

    def put_job_file(self, submission_id, name, data):
        self.files[submission_id, name] = data
        return f"store:{submission_id}/{name}"

    def read_job_file(self, submission_id, name):
        try:
            return self.files[submission_id, name]
        except KeyError as exc:
            raise FileNotFoundError(name) from exc

    def lock(self, *args):
        return SimpleNamespace(context=contextlib.nullcontext)

    def get_temp_dir(self):
        return Path(".cache/misen/test")

    def get_job_log(self, job_id, work_unit):
        return self.get_temp_dir() / f"{job_id}.log"


class ImmediateExecutor:
    def __init__(self, **kwargs):
        pass

    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **kwargs):
        pass


class Sky:
    Resources = SimpleNamespace
    Task = SimpleNamespace

    def __init__(self):
        self.requests = {}
        self.launches = []
        self.executions = []
        self.downs = []
        self.cancellations = []
        self.statuses = {}
        self.messages = []
        self.server = SimpleNamespace(common=SimpleNamespace(is_api_server_local=lambda: False))
        self.client = SimpleNamespace(sdk=self)

    def _request(self, value):
        request = f"request-{len(self.requests)}"
        self.requests[request] = value
        return request

    def get(self, request):
        value = self.requests[request]
        if isinstance(value, Exception):
            raise value
        return value

    def stream_and_get(self, request, output_stream):
        output_stream.write(f"SkyPilot request {request}\n")
        return self.get(request)

    def launch(self, task, **kwargs):
        self.launches.append((task, kwargs))
        return self._request((1, None))

    def exec(self, task, cluster_name):
        native_id = len(self.executions) + 1
        self.executions.append((task, cluster_name, native_id))
        self.statuses[cluster_name, native_id] = "RUNNING"
        return self._request((native_id, None))

    def job_status(self, cluster, job_ids):
        return self._request({job_id: self.statuses[cluster, job_id] for job_id in job_ids})

    def down(self, cluster):
        self.downs.append(cluster)
        return self._request(None)

    def cancel(self, cluster, job_ids):
        self.cancellations.append((cluster, job_ids))
        return self._request(None)

    def finish(self, name, status="SUCCEEDED"):
        _, cluster, native_id = next(row for row in self.executions if row[0].name == f"misen-{name}")
        self.events.put(
            (
                cluster,
                0,
                {"kind": "finished", "job_id": native_id, "code": 0 if status == "SUCCEEDED" else 1, "seconds": 1},
            )
        )


def spec(name, dependencies=(), **resources):
    return WorkSpec(name, f"run-{name}", aggregate_resources([resources]), list(dependencies))


def offering(index=0, *, device_memory=None, **kwargs):
    worker = SkyPilotWorker(**({"cpus": 4, "memory": 32, "max_workers": 2} | kwargs))
    return WorkerOffering(index, worker, {"cpus": worker.cpus}, device_memory)


def test_checkpoints_exclude_worker_connections_and_runtime_state():
    worker = WorkerState("worker", 0, state="ready", idle_since=123)
    worker.future = Future()
    worker.connections = [object()]
    worker.ips = ["10.0.0.1"]
    worker.prepared.add("environment")
    worker.preparing = ("prepare-job", "another-environment")
    worker.completions["job"] = {0: 0}
    checkpoint = msgspec.json.encode(ControllerState({"job": WorkState()}, workers=[worker]))
    restored = msgspec.json.decode(checkpoint, type=ControllerState).workers[0]
    assert msgspec.json.decode(checkpoint)["workers"] == [
        {"name": "worker", "offering": 0, "state": "ready", "idle_since": 123}
    ]
    assert restored.future is None and restored.preparing is None
    assert not (restored.connections or restored.ips or restored.prepared or restored.completions)
    restored.prepared.add("new-environment")
    assert "new-environment" not in worker.prepared


@pytest.mark.parametrize(
    "acknowledgements, quiesced", [([], False), ([True, True], True), ([False, True], False), ([True, False], False)]
)
def test_worker_close_requires_acknowledgement_from_every_node(acknowledgements, quiesced):
    worker = WorkerState("worker", 0)
    worker.connections = [MagicMock(close=MagicMock(return_value=ack)) for ack in acknowledgements]
    assert worker.close() is quiesced
    for connection in worker.connections:
        connection.close.assert_called_once()


def controller(monkeypatch, specs, offerings=None, store=None, sky=None, max_workers=2, **options):
    import tempfile

    sky = sky or Sky()

    def connect(handle, name, events, directory):
        sky.events = events

        def send(message):
            if message["kind"] == "run":
                task = SimpleNamespace(
                    name="misen-" + message["job_id"],
                    resources=SimpleNamespace(
                        accelerators={"GPU": len(message["env"]["CUDA_VISIBLE_DEVICES"].split(","))}
                    ),
                )
                sky.executions.append((task, name, message["job_id"]))
                sky.messages.append(message)
            elif message["kind"] == "cancel":
                sky.cancellations.append((name, message["job_id"]))

        connection = SimpleNamespace(
            ready=False, send=send, close=lambda: True, last_seen=controller_module.time.monotonic()
        )
        events.put((name, 0, {"kind": "ready", "cpus": list(range(32))}))
        return [connection], ["127.0.0.1"]

    original_dispatch = Controller._dispatch

    def dispatch(self, worker, spec, *, prepare=False):
        if prepare:
            worker.prepared.add(spec.environment_key)
            return original_dispatch(self, worker, spec)
        return original_dispatch(self, worker, spec)

    monkeypatch.setattr(Controller, "_dispatch", dispatch)
    monkeypatch.setattr(controller_module, "_connect", connect)
    monkeypatch.setattr(controller_module, "ThreadPoolExecutor", ImmediateExecutor)
    return Controller(
        sky=sky,
        workspace=store or Store(),
        submission_id="ABC",
        specs=specs,
        offerings=offerings or [offering()],
        config=SkyPilotExecutor(
            workers=[{"cpus": 4, "memory": 32}],
            max_workers=max_workers,
            lookahead_seconds=options.pop("lookahead_seconds", 0),
            reuse_workers=options.pop("reuse_workers", False),
        ),
        directory=Path(tempfile.mkdtemp()),
        stop_requested=options.pop("stop_requested", lambda: False),
        **options,
    )


def test_list_configuration_and_validation():
    config = tomllib.loads("""
        [executor]
        max_workers = 6
        [[executor.workers]]
        cpus = 16
        memory = 64
        max_workers = 4
        [[executor.workers]]
        cpus = 16
        memory = 64
        accelerators = { L4 = 1 }
        max_workers = 2
    """)
    executor = SkyPilotExecutor(**config["executor"])
    assert len(executor.workers) == 2
    assert executor.workers[1].accelerators == {"L4": 1}
    assert executor.max_workers == 6
    for kwargs in ({"workers": []}, {"max_workers": 0}, {"idle_timeout_minutes": True}):
        with pytest.raises(ValueError):
            SkyPilotExecutor(**kwargs)
    for kwargs in (
        {"cpus": 0},
        {"nodes": 0},
        {"memory": -1},
        {"accelerators": {"L4": 0.5}},
        {"accelerators": {"L4": 1, "H100": 1}},
        {"max_hourly_cost": float("nan")},
    ):
        with pytest.raises(ValueError):
            SkyPilotWorker(**({"cpus": 4, "memory": 32} | kwargs))


def test_catalog_pins_memory_variants_and_does_not_sum_device_memory():
    sky = Sky()
    records = [
        SimpleNamespace(
            region="us-east-1",
            instance_type=instance,
            accelerator_count=2,
            cpu_count=16,
            memory=128,
            device_memory=memory,
            price=price,
            spot_price=price,
        )
        for instance, memory, price in [("small", 24, 1), ("large", 80, 2)]
    ]
    sky.list_accelerators = MagicMock(return_value=sky._request({"GPU": records}))
    worker = SkyPilotWorker(cpus=8, memory=64, accelerators={"GPU": 2}, infra="aws/us-east-1")
    choices = resolve_workers(sky, [worker], {})
    request = spec("gpu", accelerators=2, accelerator_memory=40).resources
    assert not choices[0].fits(request)
    assert choices[1].fits(request)
    assert choices[1].resource_args["instance_type"] == "large"
    assert choices[1].resource_args["infra"] == "aws/us-east-1"
    # Metadata cannot inflate an offering whose actual capacity is known.
    assert resolve_workers(sky, [worker], {"GPU": 100})[0].device_memory == 24


def test_unknown_device_capacity_requires_explicit_metadata():
    worker = SkyPilotWorker(cpus=8, memory=64, infra="ssh/private", accelerators={"GPU": 2})
    request = spec("gpu", accelerators=2, accelerator_memory=40).resources
    assert not resolve_workers(Sky(), [worker], {})[0].fits(request)
    assert resolve_workers(Sky(), [worker], {"GPU": 80})[0].fits(request)
    request["accelerator_type"] = "rocm"
    assert not resolve_workers(Sky(), [worker], {"GPU": 80})[0].fits(request)


def test_real_catalog_keeps_gpu_memory_when_cpu_rows_are_present():
    pytest.importorskip("sky")
    pd = pytest.importorskip("pandas")
    from sky.catalog.common import list_accelerators_impl

    cpu = dict(
        InstanceType="m6i.xlarge",
        AcceleratorName=None,
        AcceleratorCount=0,
        vCPUs=4,
        MemoryGiB=16,
        GpuInfo=float("nan"),
        Price=0.192,
        SpotPrice=0.1,
        Region="us-east-1",
    )
    gpu = cpu | dict(
        InstanceType="g4dn.2xlarge",
        AcceleratorName="T4",
        AcceleratorCount=1,
        vCPUs=8,
        MemoryGiB=32,
        GpuInfo="{'Gpus': [{'MemoryInfo': {'SizeInMiB': 16384}}]}",
    )
    sky = Sky()

    def catalog(**kwargs):
        cloud = kwargs.pop("clouds")
        return sky._request(list_accelerators_impl(cloud, pd.DataFrame([cpu, gpu]), region_filter=None, **kwargs))

    sky.list_accelerators = catalog
    worker = SkyPilotWorker(
        cpus=4, memory=16, accelerators={"T4": 1}, infra="aws/us-east-1", instance_type="g4dn.2xlarge"
    )
    choices = resolve_workers(sky, [worker], {})
    assert len(choices) == 1
    assert choices[0].device_memory == 16
    assert choices[0].fits(spec("gpu", cpus=4, memory=16, accelerators=1, accelerator_memory=8).resources)


def test_catalog_override_cannot_bypass_incompatible_instance():
    sky = Sky()
    record = SimpleNamespace(region="us-east-1", instance_type="tiny", accelerator_count=2, cpu_count=2, memory=8)
    sky.list_accelerators = MagicMock(return_value=sky._request({"GPU": [record]}))
    worker = SkyPilotWorker(cpus=8, memory=64, accelerators={"GPU": 2})
    with pytest.raises(ConfigError, match="No catalog offering"):
        resolve_workers(sky, [worker], {"GPU": 80})


def test_preflight_accepts_local_api_and_rejects_unsatisfied_capacity(monkeypatch):
    workspace = _remote_workspace()
    graph, units = _diamond_graph()
    monkeypatch.setattr(skypilot_module, "_load_skypilot", lambda: Sky())
    executor = SkyPilotExecutor(workers=[{"cpus": 4, "memory": 32}])
    executor._validate_submission(work_graph=graph, pending_work_units=units, workspace=workspace)
    with pytest.raises(ConfigError, match="No worker type"):
        controller(monkeypatch, [spec("too-large", memory=100)])


def test_diamond_reuses_one_vm_and_packs_parallel_branches(monkeypatch):
    c = controller(monkeypatch, [spec("a"), spec("b", ["a"]), spec("c", ["a"]), spec("d", ["b", "c"])], max_workers=1)
    c.tick()  # provision just one worker for the ready root
    assert len(c.sky.launches) == 1
    assert not c.sky.executions
    c.tick()
    assert [task.name for task, _, _ in c.sky.executions] == ["misen-a"]
    c.sky.finish("a")
    c.tick()
    assert [task.name for task, _, _ in c.sky.executions] == ["misen-a", "misen-b", "misen-c"]
    c.sky.finish("b")
    c.tick()
    assert len(c.sky.executions) == 3
    c.sky.finish("c")
    c.tick()
    assert c.sky.executions[-1][0].name == "misen-d"
    c.sky.finish("d")
    c.tick()
    assert c.tick()
    assert len(c.sky.launches) == 1
    assert len(c.sky.downs) == 1


def test_provisioning_capacity_and_per_type_limits(monkeypatch):
    c = controller(monkeypatch, [spec(str(i), cpus=2, memory=16) for i in range(10)], max_workers=5)
    c.tick()
    assert len(c.sky.launches) == 2  # per-type max, two slots per starting VM
    c.tick()
    assert len(c.sky.executions) == 4
    for worker in c.state.workers:
        assert len(c._active(worker)) == 2
    c.tick()
    assert len(c.sky.launches) == 2


def test_gpu_whole_device_reservations(monkeypatch):
    o = offering(accelerators={"GPU": 2}, device_memory=80)
    c = controller(
        monkeypatch, [spec(str(i), accelerators=1, accelerator_memory=40) for i in range(3)], [o], max_workers=1
    )
    c.tick()
    c.tick()
    assert len(c.sky.executions) == 2
    c.sky.finish("0")
    c.tick()
    assert len(c.sky.executions) == 3
    assert all(task.resources.accelerators == {"GPU": 1} for task, _, _ in c.sky.executions)


def test_failure_propagates_without_running_descendants(monkeypatch):
    c = controller(monkeypatch, [spec("a"), spec("b", ["a"]), spec("independent")])
    c.tick()
    c.tick()
    c.sky.finish("a", "FAILED")
    c.tick()
    assert c.state.jobs["b"].state == "failed"
    assert c.state.jobs["independent"].state == "running"
    assert all(task.name != "misen-b" for task, _, _ in c.sky.executions)


def test_provision_failure_is_bounded_and_cleans_up(monkeypatch):
    c = controller(monkeypatch, [spec("a")])
    c.tick()
    failure = Future()
    failure.set_exception(RuntimeError("no capacity"))
    c.state.workers[0].future = failure
    with pytest.raises(RuntimeError, match="no capacity"):
        c.run()
    assert len(c.sky.launches) == 1
    assert len(c.sky.downs) == 1
    assert c.state.jobs["a"].state == "failed"


def test_storage_failure_still_cleans_up_compute(monkeypatch):
    c = controller(monkeypatch, [spec("a")])
    c.tick()
    c.workspace.put_job_file = MagicMock(side_effect=OSError("store offline"))
    with pytest.raises(OSError, match="store offline"):
        c.run()
    assert len(c.sky.downs) == 1


def test_idle_cpu_worker_yields_global_slot_for_gpu_stage(monkeypatch):
    specs = [spec("a"), spec("b", ["a"], accelerators=1), spec("c", ["b"])]
    c = controller(monkeypatch, specs, [offering(0), offering(1, accelerators={"GPU": 1})], max_workers=1)
    c.tick()
    c.tick()
    c.sky.finish("a")
    c.tick()
    assert c.state.workers[0].state == "draining"
    c.tick()
    assert len(c.sky.launches) == 2
    assert c.state.workers[1].offering == 1


def test_committed_result_wins_over_backend_failure(monkeypatch):
    c = controller(monkeypatch, [spec("a"), spec("b", ["a"])])
    c.tick()
    c.tick()
    c.workspace.put_job_file("ABC", "a.state", b"done")
    c.sky.finish("a", "FAILED")
    c.tick()
    assert c.state.jobs["a"].state == "done"
    assert c.sky.executions[-1][0].name == "misen-b"


def test_idle_cpu_capacity_is_kept_when_gpu_is_starting_or_at_type_limit(monkeypatch):
    specs = [
        spec("a", cpus=4),
        spec("b", cpus=4),
        spec("g1", ["a"], cpus=4, accelerators=1),
        spec("g2", ["b"], cpus=4, accelerators=1),
        spec("analysis", ["g1", "g2"], cpus=4),
    ]
    c = controller(
        monkeypatch,
        specs,
        [offering(0, max_workers=2), offering(1, accelerators={"GPU": 1}, max_workers=1)],
        max_workers=3,
    )
    c.tick()
    c.tick()
    c.sky.finish("a")
    c.tick()  # GPU capacity is provisioning and virtually reserves g1.
    assert len(c.sky.launches) == 3
    assert not c.sky.downs
    c.tick()
    c.sky.finish("b")
    c.tick()  # g2 waits for the sole allowed GPU; freeing a CPU cannot help.
    assert not c.sky.downs
    c.sky.finish("g1")
    c.tick()
    c.sky.finish("g2")
    c.tick()
    assert c.sky.executions[-1][0].name == "misen-analysis"
    assert len(c.sky.launches) == 3


def test_type_replacement_drains_only_one_slot_at_a_time(monkeypatch):
    specs = [spec(name, cpus=4) for name in ("a", "b", "c")]
    specs += [
        spec("gpu", ["a", "b", "c"], cpus=4, accelerators=1),
        spec("analysis", ["gpu"], cpus=4),
    ]
    c = controller(
        monkeypatch,
        specs,
        [offering(0, max_workers=3), offering(1, accelerators={"GPU": 1}, max_workers=3)],
        max_workers=3,
    )
    c.tick()
    c.tick()
    for name in ("a", "b", "c"):
        c.sky.finish(name)
    c.tick()
    assert len(c.sky.downs) == 1
    c.tick()
    assert len(c.sky.launches) == 4
    assert len(c.sky.downs) == 1


def test_runtime_memory_checks_assigned_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    run = MagicMock(return_value=SimpleNamespace(stdout="81920\n81920\n"))
    monkeypatch.setattr(subprocess, "run", run)
    request = spec("gpu", accelerators=2, accelerator_memory=40).resources
    verify_accelerators(request)
    assert run.call_args.args[0][-1] == "2,5"
    run.return_value.stdout = "24576\n24576\n"
    with pytest.raises(ConfigError, match="GiB/device"):
        verify_accelerators(request)


def test_reusable_job_handles_read_logical_states_and_cancel_one(monkeypatch, tmp_path):
    store = Store()
    workspace = _remote_workspace()
    workspace.get_result_hash.side_effect = CacheError("Result has not been computed.")
    workspace.read_job_file.side_effect = store.read_job_file
    workspace.put_job_file.side_effect = store.put_job_file
    session = MagicMock(active=True, directory=tmp_path)
    state = ControllerState({"a": WorkState(state="done"), "b": WorkState(state="running")})
    (tmp_path / "ABC.json").write_bytes(msgspec.json.encode(state))
    jobs = [
        SkyPilotJob(
            work_unit=WorkUnit(Task(_chain_task, value=i), set()),
            job_id=name,
            submission_id="ABC",
            log_path=tmp_path / name,
            workspace=workspace,
            session=session,
        )
        for i, name in enumerate(("a", "b"))
    ]
    assert list(SkyPilotJob.bulk_state(jobs).values()) == ["done", "running"]
    jobs[1].cancel()
    assert (tmp_path / "b.cancel").exists()
    session.active = False
    state.workers.append(WorkerState("owned-worker", 0, state="ready"))
    (tmp_path / "ABC.json").write_bytes(msgspec.json.encode(state))
    assert SkyPilotJob.bulk_state([jobs[1]])[jobs[1]] == "unknown"
    # Teardown acceptance must not erase explicitly uncertain execution.
    state.jobs["b"].state = "unknown"
    state.workers[0].state = "down"
    (tmp_path / "ABC.json").write_bytes(msgspec.json.encode(state))
    assert SkyPilotJob.bulk_state([jobs[1]])[jobs[1]] == "unknown"
    # Even after controller loss, a committed worker result is authoritative.
    store.put_job_file("ABC", "b.state", b"done")
    assert SkyPilotJob.bulk_state([jobs[1]])[jobs[1]] == "done"


def test_graph_dispatch_starts_local_session_and_reattaches_reordered_subset(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _, units = _diamond_graph()
    store = Store()
    snapshot = SimpleNamespace(submission_id="ABC", snapshot_key="snapshot")

    def prepare_job(unit, workspace, **kwargs):
        assert kwargs == {"dependency_jobs": {}, "reuse_env": True}
        job_id = f"job-{unit.root.kwargs['value']}"
        return job_id, ["python", "work.py"], {}, tmp_path / job_id

    snapshot.prepare_job = prepare_job
    starts = []

    class Session(skypilot_module.LocalSkySession):
        def start(self, **kwargs):
            self.process = SimpleNamespace(status=SimpleNamespace(is_active=True))
            starts.append(self)

    monkeypatch.setattr(skypilot_module, "LocalSkySession", Session)
    executor = SkyPilotExecutor(workers=[{"cpus": 4, "memory": 32}])
    jobs = {}
    executor._dispatch_work_graph(
        pending_work_units=units, jobs=jobs, workspace=store, snapshot=snapshot, progress=lambda n: None
    )
    assert len(jobs) == 4
    assert len(starts) == 1
    payload = cloudpickle.loads((starts[0].directory / "controller.pkl").read_bytes())
    assert payload["config"]._pool is None
    assert payload["config"]._sessions == {}
    specs = payload["specs"]
    assert specs[-1].dependencies == [jobs[d].job_id for d in units[-1].dependencies]
    assert "60m" in specs[1].command
    more_jobs = {units[0]: CompletedJob(units[0])}
    executor._dispatch_work_graph(
        pending_work_units=list(reversed(units[1:])),
        jobs=more_jobs,
        workspace=store,
        snapshot=snapshot,
        progress=lambda n: None,
    )
    assert len(starts) == 1
    assert all(more_jobs[w].job_id == jobs[w].job_id for w in units[1:])
    starts[0].process.status.is_active = False
    with pytest.raises(skypilot_module.SubmissionError, match="unconfirmed execution or cleanup"):
        executor._dispatch_work_graph(
            pending_work_units=units, jobs={}, workspace=store, snapshot=snapshot, progress=lambda n: None
        )
    assert len(starts) == 1


def test_sdk_reusable_interfaces_are_available():
    sky = pytest.importorskip("sky")
    for name in ("launch", "down", "list_accelerators"):
        assert callable(getattr(sky, name))
    resource = sky.Resources(cpus=2, memory=8, accelerators={"L4": 1})
    assert resource.accelerators == {"L4": 1}
    inspect.signature(sky.client.sdk.list_accelerators).bind(
        gpus_only=False,
        name_filter="^L4$",
        quantity_filter=1,
        clouds="aws",
        all_regions=True,
        require_price=False,
    )
    assert sky.client.sdk.list_accelerators is not sky.list_accelerators
    inspect.signature(sky.launch).bind(
        sky.Task(),
        cluster_name="test",
        idle_minutes_to_autostop=10,
        down=True,
    )


def test_cooperative_controller_shutdown_quiesces_before_failure(monkeypatch):
    c = controller(monkeypatch, [spec("a"), spec("b", ["a"])])
    c.tick()
    c.tick()
    c.tick()  # observe execution acceptance
    assert c.state.jobs["a"].state == "running"
    c.stop_requested = lambda: True
    with pytest.raises(ExecutionError, match="shutting down"):
        c.run()
    assert c.sky.downs
    assert all(w.state == "down" for w in c.state.workers)
    assert all(j.state == "failed" for j in c.state.jobs.values())


def test_close_waits_for_teardown_already_in_progress(monkeypatch):
    c = controller(monkeypatch, [spec("a")])
    c.tick()
    c.tick()
    c.sky.finish("a")
    c.tick()
    assert c.state.workers[0].state == "draining"
    assert len(c.sky.downs) == 1
    c.stop_requested = lambda: True
    c.run()
    assert len(c.sky.downs) == 1
    assert c.state.workers[0].state == "down"
    assert c.state.jobs["a"].state == "done"
    assert c.state.failure is None


def test_lookahead_launches_fanout_capacity_before_root_finishes(monkeypatch):
    c = controller(
        monkeypatch,
        [spec("root", cpus=4), spec("a", ["root"], cpus=4), spec("b", ["root"], cpus=4)],
        lookahead_seconds=90,
    )
    c.tick()
    assert len(c.sky.launches) == 2
    assert not c.sky.executions


def test_lookahead_never_scales_a_serial_chain(monkeypatch):
    c = controller(
        monkeypatch, [spec(str(i), [str(i - 1)] if i else [], cpus=4) for i in range(6)], lookahead_seconds=90
    )
    c.durations[""] = 1
    c.tick()
    assert len(c.sky.launches) == 1


def test_pool_reuses_workers_across_graphs_and_expires_idle_capacity(monkeypatch):
    c = controller(monkeypatch, [spec("a")], reuse_workers=True)
    c.tick()
    c.tick()
    c.sky.finish("a")
    c.tick()
    assert not c.sky.downs
    c.add_graph("SECOND", [spec("b")])
    c.tick()
    assert len(c.sky.launches) == 1
    assert c.state.jobs["a"].cluster == c.state.jobs["b"].cluster
    c.sky.finish("b")
    c.tick()
    c.state.workers[0].idle_since -= 601
    c.tick()
    c.tick()
    assert len(c.sky.downs) == 1
    assert c.state.workers[0].state == "down"
    assert (
        msgspec.json.decode(c.workspace.read_job_file("SECOND", "controller-state.json"))["jobs"]["b"]["state"]
        == "done"
    )


def test_stream_loss_never_replays_and_failed_cleanup_keeps_work_uncertain(monkeypatch):
    c = controller(monkeypatch, [spec("a"), spec("b", ["a"])])
    c.tick()
    c.tick()
    c.events.put((c.state.workers[0].name, 0, {"kind": "lost", "reason": "disconnect"}))
    c.sky.down = MagicMock(side_effect=OSError("offline"))
    with pytest.raises(ExecutionError, match="connection lost"):
        c.run()
    assert len(c.sky.executions) == 1
    assert c.state.jobs["a"].state == "running"
    assert c.state.jobs["b"].state == "failed"


def test_cancellation_waits_for_exit_event_before_freeing_reservation(monkeypatch):
    c = controller(monkeypatch, [spec("a", cpus=4), spec("b", cpus=4)], max_workers=1)
    c.tick()
    c.tick()
    (c.directory / "a.cancel").touch()
    c.tick()
    assert c.state.jobs["a"].state == "running"
    assert len(c.sky.executions) == 1
    c.sky.finish("a", "FAILED")
    c.tick()
    assert c.state.jobs["a"].state == "failed"
    assert c.state.jobs["b"].state == "running"


def test_early_agent_event_waits_for_all_connections(monkeypatch):
    c = controller(monkeypatch, [spec("a")])
    c.tick()
    future = c.state.workers[0].future
    c.state.workers[0].future = Future()
    c.tick()  # fake reader already sent ready, but the launch future is still pending
    assert not c.sky.executions
    c.state.workers[0].future = future
    c.tick()
    assert len(c.sky.executions) == 1


def test_rejected_graph_records_safe_failure_without_disrupting_pool(monkeypatch):
    c = controller(monkeypatch, [spec("a")], reuse_workers=True)
    c.tick()
    c.tick()
    (c.directory / "graph-REJECTED.pkl").write_bytes(cloudpickle.dumps([spec("too-large", memory=1000)]))
    c.tick()
    assert (c.directory / "REJECTED.json.error").exists()
    state = msgspec.json.decode(c.workspace.read_job_file("REJECTED", "controller-state.json"), type=ControllerState)
    assert state.jobs["too-large"].state == "failed"
    assert not state.workers
    assert c.state.jobs["a"].state == "running"


def test_close_drains_preparing_capacity_after_pending_job_cancel(monkeypatch):
    c = controller(monkeypatch, [spec("a")], reuse_workers=True)
    c.tick()
    c.state.workers[0].preparing = ("prepare-a", "env")
    c._finish("a", "failed", "cancelled before dispatch")
    c.stop_requested = lambda: True
    c.run()
    assert len(c.sky.downs) == 1
    assert c.state.workers[0].state == "down"


def test_multinode_reservation_waits_for_every_rank(monkeypatch):
    c = controller(monkeypatch, [spec("a", nodes=2), spec("b", nodes=2)], [offering(nodes=2)], max_workers=1)
    c.tick()
    worker = c.state.workers[0]
    connections, ips = worker.future.result()
    second = SimpleNamespace(
        ready=False, send=MagicMock(), close=lambda: True, last_seen=controller_module.time.monotonic()
    )
    connections.append(second)
    ips.append("127.0.0.2")
    c.events.put((worker.name, 1, {"kind": "ready", "cpus": list(range(32))}))
    c.tick()
    assert len(c.sky.executions) == 1
    assert second.send.call_args.args[0]["env"]["SKYPILOT_NODE_RANK"] == "1"
    c.sky.finish("a")
    c.tick()
    assert c.state.jobs["a"].state == "running"
    assert c.state.jobs["b"].state == "pending"
    c.events.put((worker.name, 1, {"kind": "finished", "job_id": "a", "code": 0, "seconds": 1}))
    c.tick()
    assert c.state.jobs["a"].state == "done"
    assert c.state.jobs["b"].state == "running"


def test_cloud_teardown_acceptance_does_not_resolve_lost_agent_execution(monkeypatch):
    c = controller(monkeypatch, [spec("a")])
    c.tick()
    c.tick()
    name = c.state.workers[0].name
    c.state.workers[0].connections[0].close = lambda: False
    c.events.put((name, 0, {"kind": "lost", "reason": "disconnect"}))
    with pytest.raises(ExecutionError, match="connection lost"):
        c.run()
    assert c.state.workers[0].state == "down"
    assert c.state.jobs["a"].state == "unknown"


def test_preparation_uses_full_worker_cpu_budget(monkeypatch):
    original_dispatch = Controller._dispatch
    c = controller(monkeypatch, [spec("a", cpus=1)])
    worker = c._provision(0)
    c._poll_jobs()
    original_dispatch(c, worker, c.work["a"], prepare=True)
    message = c.sky.messages[-1]
    assert message["cpus"] == [0, 1, 2, 3]
    assert message["env"]["OMP_NUM_THREADS"] == "4"
    assert c.state.jobs["a"].state == "pending"


@pytest.mark.parametrize("launch_failed", [False, True])
def test_teardown_waits_for_an_inflight_launch(monkeypatch, launch_failed):
    c = controller(monkeypatch, [spec("a")])
    launch = Future()
    worker = WorkerState("starting-worker", 0)
    worker.future = launch
    c.state.workers.append(worker)
    connection = MagicMock(close=MagicMock(return_value=True))
    with ThreadPoolExecutor(max_workers=2) as threads:
        c.threads = threads
        c._drain(worker)
        assert not c.sky.downs
        if launch_failed:
            launch.set_exception(RuntimeError("partially provisioned"))
        else:
            launch.set_result(([connection], ["127.0.0.1"]))
        worker.future.result(timeout=5)
        assert c.sky.downs == [worker.name]
        if not launch_failed:
            connection.close.assert_called_once()
        c._poll_jobs()
        assert worker.state == "down"
