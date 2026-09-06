"""Graph coordination through real workspace records and fake capacity APIs."""
# ruff: noqa: D103, PLR2004, S101

from __future__ import annotations

import sys
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import msgspec
import pytest

import misen.executors.skypilot as graph_module
import misen.executors.skypilot as worker_module
from misen.executors.skypilot import (
    AgentWork,
    GraphCoordinator,
    GraphWork,
    RunManifest,
    SkyPilotCapacity,
    SkyPilotTaskJob,
    read_run_state,
    run_worker_agent,
)
from misen.task_metadata import Resources, aggregate_resources
from misen.workspaces.disk import DiskWorkspace

if TYPE_CHECKING:
    from collections.abc import Callable


_RUN = "coordinator-test"


def _work(job_id: str, *parents: str, profile: str = "cpu", script: str = "pass") -> GraphWork:
    return GraphWork(
        job_id,
        list(parents),
        profile,
        [sys.executable, "-c", script],
        {},
        f"logs/{job_id}.log",
        aggregate_resources([Resources(cpus=1, memory=1, time=1)]),
    )


def _write(workspace: DiskWorkspace, name: str, **record: Any) -> None:
    workspace.put_job_file(_RUN, name, msgspec.json.encode({"version": 1, "run_id": _RUN, **record}))


def _read(workspace: DiskWorkspace, name: str) -> dict[str, Any]:
    return msgspec.json.decode(workspace.read_job_file(_RUN, name))


@dataclass
class _Backend:
    worker_launches: list[str] = field(default_factory=list)
    dedicated_launches: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)

    def launch_worker(self, agent: AgentWork) -> str:
        self.worker_launches.append(agent.worker_id)
        return agent.worker_id

    def launch_dedicated(self, node: GraphWork, attempt_id: str) -> str:
        self.dedicated_launches.append(node.job_id)
        return attempt_id

    @staticmethod
    def state(_native: str) -> str:
        return "running"

    def cancel(self, native: str) -> None:
        self.cancelled.append(native)


@dataclass
class _Calls:
    """Resolve launch/cancel inline while keeping health requests pending."""

    health: list[Future[Any]] = field(default_factory=list)

    def __call__(self, function: Callable[..., Any], *args: Any) -> Future[Any]:
        future: Future[Any] = Future()
        if function.__name__ == "state":
            self.health.append(future)
            return future
        try:
            future.set_result(function(*args))
        except Exception as exc:  # noqa: BLE001 -- emulate the production future boundary
            future.set_exception(exc)
        return future


@dataclass
class _Harness:
    coordinator: GraphCoordinator
    workspace: DiskWorkspace
    backend: _Backend
    calls: _Calls

    def ready(self, worker_id: str, *, generation: str = "generation-1") -> None:
        _write(
            self.workspace, f"worker-{worker_id}.state.json", worker_id=worker_id, generation=generation, state="idle"
        )

    def prime(self) -> None:
        self.coordinator.step()
        ready: set[str] = set()
        while set(self.backend.worker_launches) - ready:
            for worker_id in set(self.backend.worker_launches) - ready:
                self.ready(worker_id)
                ready.add(worker_id)
            self.coordinator.step()

    def complete(self, worker_id: str, *, success: bool = True) -> None:
        worker = self.coordinator.allocations[worker_id]
        assert worker.attempt_id is not None
        _write(
            self.workspace,
            f"attempt-{worker.attempt_id}.result.json",
            attempt_id=worker.attempt_id,
            state="done" if success else "failed",
        )
        _write(
            self.workspace,
            f"attempt-{worker.attempt_id}.json",
            attempt_id=worker.attempt_id,
            job_id=worker.job_id,
            worker_id=worker_id,
            generation=worker.generation,
            state="done" if success else "failed",
        )
        self.ready(worker_id, generation=cast("str", worker.generation))
        self.coordinator.step()

    def handle(self, job_id: str) -> SkyPilotTaskJob:
        handle = SkyPilotTaskJob(cast("Any", None), job_id, Path(f"logs/{job_id}.log"), self.workspace, _RUN)
        handle.coordinator = self.coordinator
        return handle


def _harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nodes: list[GraphWork],
    capacity: dict[str, SkyPilotCapacity] | None = None,
    *,
    deterministic: bool = True,
    backend: _Backend | None = None,
) -> _Harness:
    monkeypatch.chdir(tmp_path)
    workspace = DiskWorkspace(directory=str(tmp_path / "workspace"))
    capacity = capacity or {"cpu": SkyPilotCapacity(pool="cpu", memory=1)}
    executor = SimpleNamespace(
        capacity=capacity,
        setup_timeout_s=10.0,
        max_run_minutes=1,
        shutdown_timeout_s=2.0,
        poll_interval_s=0.01,
    )
    agents = [
        AgentWork(f"{name}-{index}", name, f"agent-{name}-{index}", ["unused"], {}, f"logs/agent-{name}-{index}.log")
        for name, profile in capacity.items()
        if not profile.dedicated
        for index in range(profile.max_workers)
    ]
    manifest = RunManifest(_RUN, "snapshot", nodes, agents)
    backend = backend or _Backend()
    calls = _Calls()
    if deterministic:
        monkeypatch.setattr(graph_module, "_async_call", calls)
    coordinator = GraphCoordinator(cast("Any", executor), manifest, workspace, backend)
    return _Harness(coordinator, workspace, backend, calls)


def test_completed_attempt_can_be_reconciled_after_agent_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _harness(tmp_path, monkeypatch, [_work("job")])
    harness.prime()
    worker = harness.coordinator.allocations["cpu-0"]
    _write(harness.workspace, f"attempt-{worker.attempt_id}.result.json", attempt_id=worker.attempt_id, state="done")
    _write(
        harness.workspace,
        f"attempt-{worker.attempt_id}.json",
        attempt_id=worker.attempt_id,
        worker_id=worker.worker_id,
        generation=worker.generation,
        state="done",
    )
    _write(
        harness.workspace,
        "worker-cpu-0.state.json",
        worker_id=worker.worker_id,
        generation=worker.generation,
        state="stopped",
    )
    harness.coordinator.step()
    assert worker.retired
    assert worker.job_id is None
    assert harness.coordinator.graph.states["job"].state == "done"
    assert not harness.coordinator.errors


def test_many_logical_tasks_reuse_one_native_allocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _harness(tmp_path, monkeypatch, [_work(f"job-{index}") for index in range(20)])
    harness.prime()
    seen = []
    while not harness.coordinator.graph.complete:
        worker = harness.coordinator.allocations["cpu-0"]
        assert worker.job_id is not None
        seen.append(worker.job_id)
        harness.complete("cpu-0")

    assert len(seen) == len(set(seen)) == 20
    assert harness.backend.worker_launches == ["cpu-0"]
    assert not harness.backend.dedicated_launches
    assert all(not future.done() for future in harness.calls.health)
    assert read_run_state(harness.workspace, _RUN).status == "running"
    harness.coordinator.run()
    assert read_run_state(harness.workspace, _RUN).status == "done"
    assert harness.backend.cancelled == ["cpu-0"]


def test_reversed_diamond_fanout_never_assigns_descendants_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(
        tmp_path,
        monkeypatch,
        [_work("join", "left", "right"), _work("left", "root"), _work("right", "root"), _work("root")],
        {"cpu": SkyPilotCapacity(pool="cpu", memory=1, max_workers=2)},
    )
    harness.prime()
    graph = harness.coordinator.graph
    assert graph.active == {"root"}
    assert harness.backend.worker_launches == ["cpu-0"]
    harness.complete("cpu-0")
    assert graph.states["join"].attempt_id is None
    assert harness.backend.worker_launches == ["cpu-0", "cpu-1"]
    harness.ready("cpu-1")
    harness.coordinator.step()
    assert graph.active == {"left", "right"}
    harness.complete("cpu-0")
    assert graph.states["join"].attempt_id is None
    harness.complete("cpu-1")
    assert graph.active == {"join"}
    join_worker = next(
        worker.worker_id for worker in harness.coordinator.allocations.values() if worker.job_id == "join"
    )
    harness.complete(join_worker)
    assert graph.complete


def test_durable_result_releases_successor_before_agent_cleanup_or_native_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(
        tmp_path,
        monkeypatch,
        [_work("parent"), _work("child", "parent", profile="gpu")],
        {"cpu": SkyPilotCapacity(pool="cpu", memory=1), "gpu": SkyPilotCapacity(pool="gpu", memory=1)},
    )
    harness.prime()
    worker = harness.coordinator.allocations["cpu-0"]
    _write(harness.workspace, f"attempt-{worker.attempt_id}.result.json", attempt_id=worker.attempt_id, state="done")
    harness.coordinator.step()

    assert harness.coordinator.graph.states["parent"].state == "done"
    assert worker.job_id == "parent"  # Process cleanup has not released this slot.
    assert harness.backend.worker_launches == ["cpu-0", "gpu-0"]
    harness.ready("gpu-0")
    harness.coordinator.step()
    assert harness.coordinator.graph.active == {"child"}
    assert all(not future.done() for future in harness.calls.health)


def test_post_commit_process_failure_is_reported_without_discarding_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, [_work("task")])
    harness.prime()
    worker = harness.coordinator.allocations["cpu-0"]
    _write(harness.workspace, f"attempt-{worker.attempt_id}.result.json", attempt_id=worker.attempt_id, state="done")
    harness.coordinator.step()
    assert harness.coordinator.graph.states["task"].state == "done"
    _write(
        harness.workspace,
        f"attempt-{worker.attempt_id}.json",
        attempt_id=worker.attempt_id,
        job_id="task",
        worker_id="cpu-0",
        generation=worker.generation,
        state="failed",
        reason="Log finalization failed after the callable committed its result.",
    )
    harness.ready("cpu-0", generation=cast("str", worker.generation))
    harness.coordinator.step()

    assert harness.coordinator.graph.states["task"].state == "done"
    assert harness.coordinator.errors
    assert read_run_state(harness.workspace, _RUN).cleanup_errors
    assert read_run_state(harness.workspace, _RUN).status == "running"
    harness.coordinator.run()
    assert read_run_state(harness.workspace, _RUN).status == "failed"


def test_failed_native_health_rechecks_racing_durable_success_before_failing_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, [_work("task"), _work("dependent", "task")])
    harness.prime()
    worker = harness.coordinator.allocations["cpu-0"]
    result_name = f"attempt-{worker.attempt_id}.result.json"
    original_read = DiskWorkspace.read_job_file
    result_reads = 0

    def racing_commit(workspace: DiskWorkspace, run_id: str, name: str) -> bytes:
        nonlocal result_reads
        if name != result_name:
            return original_read(workspace, run_id, name)
        result_reads += 1
        if result_reads == 1:
            raise FileNotFoundError(name)
        return msgspec.json.encode({"version": 1, "run_id": _RUN, "attempt_id": worker.attempt_id, "state": "done"})

    monkeypatch.setattr(DiskWorkspace, "read_job_file", racing_commit)
    assert worker.health is not None
    worker.health.set_result("failed")
    harness.coordinator.step()

    assert result_reads >= 2
    assert harness.coordinator.graph.states["task"].state == "done"
    assert harness.coordinator.errors  # Native process cleanup still failed, despite committed callable success.


def test_cancel_harvests_accepted_launch_before_first_status_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, [_work("task")])
    harness.coordinator.step()
    worker = harness.coordinator.allocations["cpu-0"]
    assert worker.native is None
    assert worker.launch.done()
    harness.coordinator.cancelled.set()
    harness.coordinator.run()

    assert harness.coordinator.finished.is_set()
    assert harness.backend.cancelled == ["cpu-0"]
    assert not harness.coordinator.errors
    assert _read(harness.workspace, "worker-cpu-0.lease.json")["stop"] is True


def test_stale_generation_cannot_commit_and_changed_generation_is_not_readmitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, [_work("first"), _work("unrelated")])
    harness.prime()
    worker = harness.coordinator.allocations["cpu-0"]
    first_attempt = worker.attempt_id
    _write(
        harness.workspace,
        f"attempt-{first_attempt}.json",
        attempt_id=first_attempt,
        worker_id="cpu-0",
        generation="obsolete-generation",
        state="done",
    )
    harness.coordinator.step()
    assert harness.coordinator.graph.states["first"].state == "pending"
    harness.ready("cpu-0", generation="generation-2")
    harness.coordinator.step()
    assert harness.coordinator.graph.states["first"].state == "failed"
    assert harness.coordinator.graph.states["first"].attempt_id == first_attempt
    assert worker.retired
    assert harness.backend.cancelled == ["cpu-0"]
    assert harness.coordinator.graph.states["unrelated"].attempt_id is None
    _write(harness.workspace, f"attempt-{first_attempt}.result.json", attempt_id=first_attempt, state="done")
    harness.coordinator.step()
    assert harness.coordinator.graph.states["first"].state == "failed"
    assert harness.coordinator.graph.states["unrelated"].attempt_id is None
    assert harness.backend.worker_launches == ["cpu-0"]


def test_callable_failure_does_not_block_independent_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _harness(tmp_path, monkeypatch, [_work("bad"), _work("child", "bad"), _work("good")])
    harness.prime()
    harness.complete("cpu-0", success=False)
    graph = harness.coordinator.graph
    assert graph.states["bad"].state == graph.states["child"].state == "failed"
    assert graph.states["child"].attempt_id is None
    assert harness.coordinator.allocations["cpu-0"].job_id == "good"
    harness.complete("cpu-0")
    assert graph.states["good"].state == "done"
    assert graph.complete


@pytest.mark.parametrize("failure", ["launch", "bootstrap-ended", "bootstrap-timeout"])
def test_one_failed_allocation_does_not_fail_work_while_a_compatible_worker_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    harness = _harness(
        tmp_path,
        monkeypatch,
        [_work("first"), _work("second")],
        {"cpu": SkyPilotCapacity(pool="cpu", memory=1, max_workers=2)},
    )
    harness.coordinator.step()
    harness.ready("cpu-0")
    harness.coordinator.step()
    failed_worker = harness.coordinator.allocations["cpu-1"]
    if failure == "launch":
        failed_launch: Future[Any] = Future()
        failed_launch.set_exception(RuntimeError("Capacity could not launch."))
        failed_worker.launch = failed_launch
    else:
        harness.coordinator.step()
        if failure == "bootstrap-ended":
            assert failed_worker.health is not None
            failed_worker.health.set_result("failed")
        else:
            failed_worker.started_at -= harness.coordinator.executor.setup_timeout_s + 1
    harness.coordinator.step()

    graph = harness.coordinator.graph
    assert failed_worker.retired
    assert graph.active == {"first"}
    assert graph.states["second"].state == "pending"
    harness.complete("cpu-0")
    assert harness.coordinator.allocations["cpu-0"].job_id == "second"
    harness.complete("cpu-0")
    assert graph.complete
    assert all(state.state == "done" for state in graph.states.values())


def test_cancel_one_task_does_not_fail_unrelated_work_on_the_same_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, [_work("cancelled"), _work("dependent", "cancelled"), _work("unrelated")])
    harness.prime()
    harness.handle("cancelled").cancel()
    harness.coordinator.step()

    graph = harness.coordinator.graph
    assert graph.states["cancelled"].state == "failed"
    assert graph.states["dependent"].state == "failed"
    assert graph.states["dependent"].attempt_id is None
    assert graph.states["unrelated"].state == "pending"
    assert not harness.backend.cancelled
    lease = _read(harness.workspace, "worker-cpu-0.lease.json")
    assert lease["stop"] is False
    assert lease["cancel_attempt_id"] == graph.states["cancelled"].attempt_id
    harness.complete("cpu-0", success=False)
    assert harness.coordinator.allocations["cpu-0"].job_id == "unrelated"
    harness.complete("cpu-0")
    assert graph.states["unrelated"].state == "done"
    assert harness.backend.worker_launches == ["cpu-0"]


def test_cancel_does_not_touch_another_profiles_capacity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _harness(
        tmp_path,
        monkeypatch,
        [_work("cancelled"), _work("unrelated", profile="other")],
        {"cpu": SkyPilotCapacity(pool="cpu", memory=1), "other": SkyPilotCapacity(cluster="other", memory=1)},
    )
    harness.prime()
    harness.handle("cancelled").cancel()
    harness.coordinator.step()

    assert not harness.coordinator.allocations["other-0"].retired
    assert harness.coordinator.graph.states["unrelated"].state == "pending"
    assert _read(harness.workspace, "worker-other-0.lease.json")["stop"] is False
    assert "other-0" not in harness.backend.cancelled
    harness.complete("other-0")
    assert harness.coordinator.graph.states["unrelated"].state == "done"


def test_cancel_before_agent_admission_never_starts_the_cancelled_callable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, [_work("cancelled"), _work("unrelated")])
    harness.prime()
    worker = harness.coordinator.allocations["cpu-0"]
    agent = worker_module._Agent(  # noqa: SLF001 -- control the pre-admission race deterministically
        harness.workspace,
        _RUN,
        "cpu-0",
        10,
        0.05,
        0.01,
        60,
        generation=cast("str", worker.generation),
    )
    harness.handle("cancelled").cancel()
    harness.coordinator.step()

    def forbidden_process(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("A cancellation observed before admission must prevent the task process from starting.")

    monkeypatch.setattr(worker_module.subprocess, "Popen", forbidden_process)
    assert agent._lease() is None  # noqa: SLF001 -- cancellation arrives while no subprocess exists
    agent._admit(graph_module.time.monotonic() + 5)  # noqa: SLF001 -- then the already-written command is read
    outcome = _read(harness.workspace, f"attempt-{worker.attempt_id}.json")
    assert outcome["state"] == "failed"
    assert agent.active is None


def test_idle_step_control_reads_do_not_scale_with_blocked_graph_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nodes = [_work(f"job-{index}", *([f"job-{index - 1}"] if index else [])) for index in range(1_000)]
    harness = _harness(tmp_path, monkeypatch, nodes)
    harness.prime()
    reads: list[str] = []
    original_read = DiskWorkspace.read_job_file

    def counted(workspace: DiskWorkspace, run_id: str, name: str) -> bytes:
        reads.append(name)
        return original_read(workspace, run_id, name)

    monkeypatch.setattr(DiskWorkspace, "read_job_file", counted)
    harness.coordinator.step()
    assert len(reads) <= 20, "One active worker must not trigger per-node cloud reads for a blocked 1k-node graph."


def test_unchanged_step_does_not_republish_whole_run_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _harness(tmp_path, monkeypatch, [_work("first"), _work("second", "first")])
    harness.prime()
    writes: list[str] = []
    original_write = DiskWorkspace.put_job_file

    def counted(workspace: DiskWorkspace, run_id: str, name: str, data: bytes) -> str:
        writes.append(name)
        return original_write(workspace, run_id, name, data)

    monkeypatch.setattr(DiskWorkspace, "put_job_file", counted)
    for _ in range(3):
        harness.coordinator.step()
    assert "run-state.json" not in writes


def test_run_deadline_stops_only_owned_agent_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _harness(tmp_path, monkeypatch, [_work("first"), _work("second", "first")])
    harness.prime()
    harness.coordinator.started_at -= 120
    harness.coordinator.run()

    assert harness.coordinator.finished.is_set()
    assert harness.coordinator.graph.complete
    assert harness.backend.cancelled == ["cpu-0"]
    assert _read(harness.workspace, "worker-cpu-0.lease.json")["stop"] is True
    assert all(state.state == "failed" for state in harness.coordinator.graph.states.values())
    assert read_run_state(harness.workspace, _RUN).status == "failed"


def test_dedicated_setup_timeout_retains_reservation_until_native_terminal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(graph_module.time, "monotonic", lambda: clock.now)
    harness = _harness(
        tmp_path,
        monkeypatch,
        [_work("first"), _work("second")],
        {"cpu": SkyPilotCapacity(infra="aws", memory=1, dedicated=True)},
    )
    harness.coordinator.step()
    harness.coordinator.step()
    worker = next(iter(harness.coordinator.allocations.values()))
    clock.now += 11
    harness.coordinator.step()

    graph = harness.coordinator.graph
    assert graph.states["first"].state == "failed"
    assert "setup_timeout_s" in cast("str", graph.states["first"].reason)
    assert graph.states["second"].state == "pending"
    assert harness.backend.cancelled == [worker.native]
    assert harness.backend.dedicated_launches == ["first"]
    assert not worker.retired
    assert worker.health is not None
    worker.health.set_result("failed")
    harness.coordinator.step()
    assert worker.retired
    assert harness.backend.dedicated_launches == ["first", "second"]


def test_dedicated_execution_deadline_starts_at_callable_event_not_allocation_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(graph_module.time, "monotonic", lambda: clock.now)
    harness = _harness(
        tmp_path,
        monkeypatch,
        [_work("task")],
        {"cpu": SkyPilotCapacity(infra="aws", memory=1, dedicated=True)},
    )
    harness.coordinator.step()
    worker = next(iter(harness.coordinator.allocations.values()))
    clock.now = 105.0
    _write(
        harness.workspace, f"attempt-{worker.attempt_id}.started.json", attempt_id=worker.attempt_id, state="running"
    )
    harness.coordinator.step()
    clock.now = 120.0
    harness.coordinator.step()
    assert harness.coordinator.graph.states["task"].state == "running"
    assert not harness.backend.cancelled
    clock.now = 165.0
    harness.coordinator.step()
    assert harness.coordinator.graph.states["task"].state == "failed"
    assert "execution deadline" in cast("str", harness.coordinator.graph.states["task"].reason)
    harness.coordinator.step()
    assert harness.backend.cancelled == [worker.native]


def test_timed_out_pending_dedicated_launch_is_cancelled_when_acceptance_arrives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(graph_module.time, "monotonic", lambda: clock.now)
    harness = _harness(
        tmp_path,
        monkeypatch,
        [_work("first"), _work("second")],
        {"cpu": SkyPilotCapacity(infra="aws", memory=1, dedicated=True)},
    )
    pending_launch: Future[Any] = Future()

    def delayed_launch(function: Callable[..., Any], *args: Any) -> Future[Any]:
        return pending_launch if function.__name__ == "launch_dedicated" else harness.calls(function, *args)

    monkeypatch.setattr(graph_module, "_async_call", delayed_launch)
    harness.coordinator.step()
    clock.now += 11
    harness.coordinator.step()
    assert harness.coordinator.graph.states["first"].state == "failed"
    assert harness.coordinator.graph.states["second"].attempt_id is None
    assert not harness.backend.cancelled
    pending_launch.set_result("accepted-late")
    harness.coordinator.step()
    assert harness.backend.cancelled == ["accepted-late"]
    assert harness.coordinator.graph.states["second"].attempt_id is None


def test_shutdown_deadline_is_shared_across_all_capacity_cancellations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(
        tmp_path,
        monkeypatch,
        [_work("first"), _work("second")],
        {"cpu": SkyPilotCapacity(pool="cpu", memory=1, max_workers=2)},
    )
    harness.prime()
    clock = SimpleNamespace(now=100.0)
    timeouts: list[float] = []

    class DeadlineFuture(Future[Any]):
        def result(self, timeout: float | None = None) -> Any:
            assert timeout is not None
            timeouts.append(timeout)
            clock.now += timeout
            raise TimeoutError

    def pending_cancel(function: Callable[..., Any], *_args: Any) -> Future[Any]:
        assert function.__name__ == "cancel"
        return DeadlineFuture()

    monkeypatch.setattr(graph_module, "_async_call", pending_cancel)
    monkeypatch.setattr(graph_module.time, "monotonic", lambda: clock.now)
    harness.coordinator.cancelled.set()
    harness.coordinator.run()

    assert timeouts == [2.0, 0.0]
    assert harness.coordinator.finished.is_set()
    assert len(harness.coordinator.errors) == 2
    assert all("Cleanup unresolved" in error for error in harness.coordinator.errors)


def test_live_coordinator_and_agent_finish_graph_without_any_job_status_polling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    threads: dict[str, threading.Thread] = {}
    failures: list[BaseException] = []

    class LiveBackend(_Backend):
        def launch_worker(self, agent: AgentWork) -> str:
            native = super().launch_worker(agent)

            def run_agent() -> None:
                try:
                    run_worker_agent(
                        harness.workspace,
                        _RUN,
                        agent.worker_id,
                        lease_timeout_s=3,
                        shutdown_grace_s=0.05,
                        poll_interval_s=0.01,
                        max_runtime_s=10,
                    )
                except BaseException as exc:  # noqa: BLE001 -- surface agent-thread failures to the test
                    failures.append(exc)

            threads[native] = threading.Thread(target=run_agent, daemon=True)
            threads[native].start()
            return native

        def cancel(self, native: str) -> None:
            super().cancel(native)
            _write(harness.workspace, f"worker-{native}.lease.json", worker_id=native, sequence=2**31, stop=True)
            threads[native].join(timeout=3)

    backend = LiveBackend()
    harness = _harness(tmp_path, monkeypatch, [], backend=backend, deterministic=False)
    directory = harness.workspace.get_temp_dir().parent / "job_files" / _RUN
    script = (
        "import json, os; from pathlib import Path; "
        "attempt = os.environ['MISEN_ATTEMPT_ID']; "
        "record = {'version': 1, 'run_id': os.environ['MISEN_RUN_ID'], 'attempt_id': attempt, 'state': 'done'}; "
        f"root = Path({str(directory)!r}); "
        "path = root / ('attempt-' + attempt + '.result.json'); "
        "temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(record)); temporary.replace(path)"
    )
    manifest = RunManifest(
        _RUN,
        "snapshot",
        [_work("first", script=script), _work("second", "first", script=script)],
        harness.coordinator.manifest.agents,
    )
    harness.coordinator = GraphCoordinator(harness.coordinator.executor, manifest, harness.workspace, backend)
    harness.coordinator.start()
    try:
        assert harness.coordinator.finished.wait(5), (
            "Coordinator must make progress without calling any logical-job state method."
        )
        assert harness.coordinator.snapshot_state().status == "done"
        assert not harness.coordinator.errors
        assert backend.worker_launches == ["cpu-0"]
        assert not failures
    finally:
        harness.coordinator.close()
        for native, thread in threads.items():
            _write(harness.workspace, f"worker-{native}.lease.json", worker_id=native, sequence=2**31 + 1, stop=True)
            thread.join(timeout=3)
            assert not thread.is_alive()
