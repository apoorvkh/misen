# ruff: noqa: ANN001, ANN003, ANN201, ANN202, D100, D103, PLR2004, S101, SLF001
import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest

import misen.executors.skypilot as graph_mod
import misen.executors.skypilot as sky_mod
from misen.exceptions import ConfigError, ExecutionError, StatusQueryError, StorageError, SubmissionError
from misen.executors.skypilot import AgentWork, GraphWork, RunManifest, SkyPilotCapacity


class _RequestId(str):
    __slots__ = ()


class _Store:
    def __init__(self) -> None:
        self.files: dict[tuple[str, str], bytes] = {}
        self.writes: list[dict] = []
        self.fail_write: int | None = None

    def put_job_file(self, run_id, name, data):
        if self.fail_write == len(self.writes):
            msg = "storage unavailable"
            raise StorageError(msg)
        self.files[run_id, name] = data
        self.writes.append(msgspec.json.decode(data))

    def read_job_file(self, run_id, name):
        try:
            return self.files[run_id, name]
        except KeyError:
            raise FileNotFoundError(name) from None

    def lock(self, _namespace, _key):
        def context(*, timeout):
            assert timeout > 0
            return contextlib.nullcontext()

        return SimpleNamespace(context=context)


@pytest.fixture
def setup(monkeypatch):
    store = _Store()
    responses = {
        "cluster-request": (17, None),
        "managed-request": ([29], None),
        "status-request": {"17": "RUNNING"},
        "cancel-request": None,
        "queue-request": [],
    }

    def get(request_id):
        result = responses[request_id]
        if isinstance(result, Exception):
            raise result
        return result

    sky = SimpleNamespace(
        Resources=Mock(side_effect=lambda **options: SimpleNamespace(options=options)),  # noqa: PLW0108
        Task=Mock(side_effect=lambda **options: SimpleNamespace(options=options)),  # noqa: PLW0108
        exec=Mock(return_value=_RequestId("cluster-request")),
        jobs=SimpleNamespace(launch=Mock(return_value=_RequestId("managed-request"))),
        get=Mock(side_effect=get),
        job_status=Mock(return_value="status-request"),
        cancel=Mock(return_value="cancel-request"),
        queue=Mock(return_value="queue-request"),
        server=SimpleNamespace(common=SimpleNamespace(is_api_server_local=Mock(return_value=False))),
    )

    class ManagedJob:
        def __init__(self, **values) -> None:
            self.__dict__.update(values)
            self.label = "managed allocation"
            self.cancelled = False
            self._api_session = getattr(sky, "test_session", None)
            if self._api_session is not None:
                self._api_session.jobs.append(self)

        def _resolve_managed_job_id(self, client):
            self.managed_job_id = client.get(self.request_id)[0][0]
            return self.managed_job_id

        def state(self):
            self._resolve_managed_job_id(sky)
            return "running"

        def cancel(self):
            self._resolve_managed_job_id(sky)
            self.cancelled = True

    monkeypatch.setattr(sky_mod, "_load_skypilot", lambda: sky)
    monkeypatch.setattr(sky_mod, "SkyPilotJob", ManagedJob)
    executor = SimpleNamespace(
        capacity={"cpu": SkyPilotCapacity(cluster="existing-cpu", cpus=4, memory=8)},
        coordinator=SkyPilotCapacity(infra="aws/us-east-1", dedicated=True, cpus=2, memory=4),
        name_prefix="misen",
        max_run_minutes=60,
        setup_timeout_s=120,
    )
    manifest = RunManifest("run-id", "snapshot-key", [], [])
    backend = graph_mod._SkyCapacityBackend(executor, manifest, store)
    agent = AgentWork("worker-id", "cpu", "job-id", ["python", "worker.py"], {}, "logs/job.log")
    return backend, agent, store, sky, responses


def _allocation_record(store, allocation_id="worker-id"):
    return msgspec.json.decode(store.files["run-id", f"allocation-{allocation_id}.json"])


def test_cluster_submission_is_recorded_before_and_after_acceptance(setup):
    backend, agent, store, sky, _responses = setup

    def accepted(task, *, cluster_name):
        record = _allocation_record(store)
        assert record["launch_state"] == "submitting"
        assert record["request_id"] is None
        assert cluster_name == "existing-cpu"
        assert task.options["resources"][0].options == {"cpus": "4+", "memory": "8+"}
        return _RequestId("cluster-request")

    sky.exec.side_effect = accepted
    native = backend.launch_worker(agent)

    record = _allocation_record(store)
    assert record["launch_state"] == "accepted"
    assert record["request_id"] == "cluster-request"
    assert type(native.request_id) is str
    assert record["native_job_id"] is None
    sky.jobs.launch.assert_not_called()
    assert sky.Task.call_args.kwargs["api_server_access"] is False


def test_cluster_status_normalizes_json_keys_and_persists_native_identity(setup):
    backend, agent, store, sky, _responses = setup
    native = backend.launch_worker(agent)

    assert backend.state(native) == "running"
    assert _allocation_record(store)["native_job_id"] == 17
    sky.job_status.assert_called_once_with(cluster_name="existing-cpu", job_ids=[17])


def test_cluster_cancel_only_targets_its_native_job(setup):
    backend, agent, store, sky, _responses = setup
    native = backend.launch_worker(agent)

    backend.cancel(native)

    sky.cancel.assert_called_once_with(cluster_name="existing-cpu", job_ids=[17])
    assert _allocation_record(store)["native_job_id"] == 17


def test_repeated_launch_reattaches_without_another_submission_or_launch_result_lookup(setup):
    backend, agent, _store, sky, responses = setup
    native = backend.launch_worker(agent)
    backend.state(native)
    responses["cluster-request"] = RuntimeError("expired")
    sky.get.reset_mock()

    restored = backend.launch_worker(agent)
    assert backend.state(restored) == "running"

    sky.exec.assert_called_once()
    assert restored.job_id == 17
    assert all(call.args != ("cluster-request",) for call in sky.get.call_args_list)


def test_reusing_allocation_with_different_shape_is_rejected(setup):
    backend, agent, _store, sky, _responses = setup
    backend.launch_worker(agent)
    backend.executor.capacity["cpu"] = SkyPilotCapacity(cluster="existing-cpu", cpus=8, memory=8)

    with pytest.raises(SubmissionError, match="different submission parameters"):
        backend.launch_worker(agent)

    sky.exec.assert_called_once()


@pytest.mark.parametrize("request_id", [None, False, 123, ""])
def test_invalid_request_identity_is_not_stringified_or_replayed(setup, request_id):
    backend, agent, store, sky, _responses = setup
    sky.exec.return_value = request_id

    with pytest.raises(SubmissionError, match="no valid allocation request identity"):
        backend.launch_worker(agent)
    assert _allocation_record(store)["request_id"] is None
    with pytest.raises(SubmissionError, match="acceptance is uncertain"):
        backend.launch_worker(agent)
    sky.exec.assert_called_once()


def test_unresolved_launch_exception_is_not_replayed(setup):
    backend, agent, _store, sky, _responses = setup
    sky.exec.side_effect = TimeoutError("acceptance uncertain")

    with pytest.raises(SubmissionError, match="failed or is uncertain"):
        backend.launch_worker(agent)
    with pytest.raises(SubmissionError, match="acceptance is uncertain"):
        backend.launch_worker(agent)

    sky.exec.assert_called_once()


def test_pre_acceptance_storage_failure_prevents_submission(setup):
    backend, agent, store, sky, _responses = setup
    store.fail_write = 0

    with pytest.raises(StorageError, match="storage unavailable"):
        backend.launch_worker(agent)

    sky.exec.assert_not_called()


def test_post_acceptance_storage_failure_retains_the_native_cancel_handle(setup):
    backend, agent, store, sky, _responses = setup
    store.fail_write = 1

    with pytest.raises(SubmissionError, match="durable request record") as caught:
        backend.launch_worker(agent)

    (native,) = caught.value.submitted_jobs
    assert native.request_id == "cluster-request"
    native.cancel()
    sky.cancel.assert_called_once_with(cluster_name="existing-cpu", job_ids=[17])


def test_cleanup_can_repair_previously_failed_acceptance_record(setup):
    backend, agent, store, sky, _responses = setup
    store.fail_write = 1
    with pytest.raises(SubmissionError, match="durable request record") as caught:
        backend.launch_worker(agent)
    store.fail_write = None
    (native,) = caught.value.submitted_jobs

    backend.cancel(native)

    assert _allocation_record(store)["request_id"] == "cluster-request"
    assert _allocation_record(store)["native_job_id"] == 17
    sky.cancel.assert_called_once_with(cluster_name="existing-cpu", job_ids=[17])


@pytest.mark.parametrize("result", [(None, None), (True, None), (0, None), (-1, None), ("17", None), [], ()])
def test_invalid_cluster_native_identity_never_broadens_cancellation(setup, result):
    backend, agent, _store, sky, responses = setup
    native = backend.launch_worker(agent)
    responses["cluster-request"] = result

    with pytest.raises(ExecutionError, match="invalid cluster job identity"):
        backend.cancel(native)

    assert native.job_id is None
    sky.cancel.assert_not_called()


def test_cluster_native_identity_persists_even_when_followup_status_fails(setup):
    backend, agent, store, _sky, responses = setup
    native = backend.launch_worker(agent)
    responses["status-request"] = OSError("status unavailable")

    with pytest.raises(OSError, match="status unavailable"):
        backend.state(native)

    assert _allocation_record(store)["native_job_id"] == 17


@pytest.mark.parametrize("as_model", [False, True])
def test_expired_cluster_request_recovers_only_exact_named_job(setup, as_model):
    backend, agent, store, sky, responses = setup
    native = backend.launch_worker(agent)
    responses["cluster-request"] = RuntimeError("expired")
    responses["queue-request"] = [
        {"job_name": "unrelated", "job_id": 99},
        {"job_name": native.name, "job_id": 17},
    ]
    if as_model:
        responses["queue-request"] = [SimpleNamespace(**record) for record in responses["queue-request"]]

    backend.cancel(native)

    sky.queue.assert_called_once_with(cluster_name="existing-cpu", skip_finished=False, all_users=False)
    sky.cancel.assert_called_once_with(cluster_name="existing-cpu", job_ids=[17])
    assert _allocation_record(store)["native_job_id"] == 17


@pytest.mark.parametrize("count", [0, 2])
def test_absent_or_ambiguous_recovery_never_cancels_unrelated_jobs(setup, count):
    backend, agent, _store, sky, responses = setup
    native = backend.launch_worker(agent)
    responses["cluster-request"] = RuntimeError("expired")
    responses["queue-request"] = [{"job_name": native.name, "job_id": number + 1} for number in range(count)]

    with pytest.raises(StatusQueryError, match="Could not recover"):
        backend.cancel(native)

    sky.cancel.assert_not_called()


@pytest.mark.parametrize("source", [{"pool": "existing-pool"}, {"infra": ["aws/us-east-1", "aws/us-west-2"]}])
def test_managed_allocation_launches_once_with_profile_and_persists_resolved_id(setup, source):
    backend, agent, store, sky, _responses = setup
    backend.executor.capacity["cpu"] = SkyPilotCapacity(**source, cpus=4, memory=8)
    native = backend.launch_worker(agent)

    assert backend.state(native) == "running"
    restored = backend.launch_worker(agent)

    sky.jobs.launch.assert_called_once()
    sky.exec.assert_not_called()
    assert sky.jobs.launch.call_args.kwargs["pool"] == source.get("pool")
    assert _allocation_record(store)["native_job_id"] == restored.managed_job_id == 29
    assert type(native.request_id) is str
    if "infra" in source:
        assert [item.options["infra"] for item in sky.Task.call_args.kwargs["resources"]] == source["infra"]


def test_graph_managed_allocations_are_not_drained_again_by_legacy_session_close(setup):
    backend, agent, _store, sky, _responses = setup
    existing_job = object()
    sky.test_session = SimpleNamespace(jobs=[existing_job])
    backend.executor.capacity["cpu"] = SkyPilotCapacity(pool="existing-pool")

    native = backend.launch_worker(agent)

    assert native._api_session is sky.test_session
    assert sky.test_session.jobs == [existing_job]


def test_detached_coordinator_acknowledgement_is_durable_and_grants_api_access(setup):
    backend, _agent, store, sky, _responses = setup

    native = backend.launch_coordinator("coordinator-job", ["python", "coordinator.py"], {}, Path("logs/control.log"))

    assert native.managed_job_id == 29
    assert _allocation_record(store, "coordinator-run-id")["native_job_id"] == 29
    assert sky.Task.call_args.kwargs["api_server_access"] is True


def test_detached_coordinator_rejects_local_api_before_provisioning(setup):
    backend, _agent, store, sky, _responses = setup
    sky.server.common.is_api_server_local.return_value = True

    with pytest.raises(ConfigError, match="stable remote"):
        backend.launch_coordinator("job", ["python"], {}, Path("logs/job.log"))

    assert store.writes == []
    sky.jobs.launch.assert_not_called()


def test_detached_unresolved_acknowledgement_retains_accepted_native_handle(setup):
    backend, _agent, _store, _sky, responses = setup
    responses["managed-request"] = ([None], None)

    with pytest.raises(SubmissionError, match="durably acknowledge") as caught:
        backend.launch_coordinator("job", ["python"], {}, Path("logs/job.log"))

    (native,) = caught.value.submitted_jobs
    assert native.request_id == "managed-request"
    assert native.managed_job_id is None


def test_dedicated_launch_reserves_profile_and_passes_attempt_environment(setup):
    backend, _agent, _store, sky, _responses = setup
    backend.executor.capacity["gpu"] = SkyPilotCapacity(
        infra="aws", cpus=8, memory=32, accelerators={"L4": 1}, dedicated=True
    )
    node = GraphWork(
        "task-job", [], "gpu", ["python", "payload.py"], {"SAFE_VALUE": "two words"}, "logs/task.log", {"time": 5}
    )

    backend.launch_dedicated(node, "attempt-id")

    options = sky.Task.call_args.kwargs
    assert options["resources"][0].options["cpus"] == "8+"
    assert options["resources"][0].options["accelerators"] == {"L4": 1}
    assert "MISEN_RUN_ID=run-id" in options["run"]
    assert "MISEN_ATTEMPT_ID=attempt-id" in options["run"]
    assert "7m bash" in options["run"]


def test_ranked_dask_command_uses_existing_runtime_wrapper():
    profile = SkyPilotCapacity(infra="aws", nodes=2, dedicated=True, cpus=2, memory=4)

    command = graph_mod._run_command(
        ["python", "payload.py"], {}, Path("logs/job.log"), time_minutes=10, profile=profile, uses_dask_client=True
    )

    assert "SKYPILOT_NODE_RANK" in command
    assert "SKYPILOT_NODE_IPS" in command
    assert "MISEN_DASK_ROLE" in command
    assert "10m bash" in command


def test_non_dask_multinode_payload_is_only_run_on_rank_zero():
    command = graph_mod._run_command(
        ["python", "payload.py"],
        {},
        Path("logs/job.log"),
        time_minutes=1,
        profile=SkyPilotCapacity(infra="aws", nodes=2, dedicated=True),
        uses_dask_client=False,
    )

    assert "${SKYPILOT_NODE_RANK:-0}" in command
    assert '!= "0"' in command
