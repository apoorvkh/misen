"""Ready-only graph admission, event fencing, and durable run-state contracts."""
# ruff: noqa: D103, S101

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import pytest

from misen.exceptions import StorageError
from misen.executors.skypilot import (
    AgentWork,
    GraphWork,
    LogicalState,
    ReadyGraph,
    RunManifest,
    RunState,
    read_run_state,
)
from misen.task_metadata import Resources
from misen.workspaces.disk import DiskWorkspace

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _work(job_id: str, *dependencies: str) -> GraphWork:
    return GraphWork(
        job_id=job_id,
        dependencies=list(dependencies),
        profile="cpu",
        argv=["python", "worker.py"],
        env={},
        log_path=f"job_logs/{job_id}.log",
        resources=Resources(cpus=1, memory=1),
    )


def _run_next(graph: ReadyGraph, worker_id: str = "worker-1") -> str:
    key = graph.ready.popleft()
    attempt = f"attempt-{key}"
    graph.assign(key, attempt, worker_id)
    assert graph.running(key, attempt)
    assert graph.apply_result(key, attempt, success=True)
    return key


def test_empty_graph_is_complete_without_capacity() -> None:
    graph = ReadyGraph([])

    assert graph.complete
    assert not graph.ready
    assert not graph.active
    assert not graph.finished


def test_reversed_chain_uses_single_worker_without_dependency_waiting() -> None:
    graph = ReadyGraph([_work("last", "middle"), _work("middle", "first"), _work("first")])

    assert list(graph.ready) == ["first"]
    assert graph.states["first"].reason == "Waiting for compatible capacity."
    assert graph.states["middle"].reason == "Waiting for dependencies."
    order = []
    while graph.ready:
        assert len(graph.ready) == 1
        order.append(_run_next(graph))

    assert order == ["first", "middle", "last"]
    assert graph.complete
    assert not graph.active
    assert all(state.state == "done" for state in graph.states.values())
    assert all(state.worker_id == "worker-1" for state in graph.states.values())


def test_diamond_releases_fanout_and_waits_for_every_join_parent() -> None:
    graph = ReadyGraph([_work("join", "left", "right"), _work("right", "root"), _work("left", "root"), _work("root")])

    assert _run_next(graph) == "root"
    assert set(graph.ready) == {"left", "right"}
    left = graph.ready.popleft()
    right = graph.ready.popleft()
    graph.assign(left, "attempt-left", "worker-left")
    graph.assign(right, "attempt-right", "worker-right")

    assert graph.apply_result(left, "attempt-left", success=True)
    assert not graph.ready
    assert graph.remaining["join"] == 1
    assert not graph.complete
    assert graph.apply_result(right, "attempt-right", success=True)
    assert list(graph.ready) == ["join"]
    assert _run_next(graph) == "join"
    assert graph.complete


def test_blocked_work_never_claims_worker_or_attempt() -> None:
    graph = ReadyGraph([_work("child", "parent"), _work("parent")])

    with pytest.raises(ValueError, match="Only a ready, unassigned work unit"):
        graph.assign("child", "wrong-attempt", "worker-1")
    assert graph.states["child"].attempt_id is None
    assert graph.states["child"].worker_id is None
    assert not graph.active

    graph.ready.popleft()
    graph.assign("parent", "parent-attempt", "worker-1")
    assert graph.states["parent"].state == "pending"
    assert graph.states["parent"].reason == "Preparing execution environment."
    with pytest.raises(ValueError, match="Only a ready, unassigned work unit"):
        graph.assign("parent", "duplicate-attempt", "worker-2")
    assert graph.states["parent"].attempt_id == "parent-attempt"
    assert graph.states["parent"].worker_id == "worker-1"


def test_failure_propagates_only_to_descendants_and_does_not_consume_workers() -> None:
    graph = ReadyGraph(
        [
            _work("bad-root"),
            _work("bad-child", "bad-root"),
            _work("good-root"),
            _work("good-child", "good-root"),
            _work("join", "bad-child", "good-child"),
            _work("final", "join"),
        ]
    )
    assert graph.ready.popleft() == "bad-root"
    graph.assign("bad-root", "attempt-bad", "worker-1")
    assert graph.apply_result("bad-root", "attempt-bad", success=False, reason="User function failed.")

    for key in ("bad-root", "bad-child", "join", "final"):
        assert graph.states[key].state == "failed"
    for key in ("bad-child", "join", "final"):
        assert graph.states[key].attempt_id is None
        assert graph.states[key].worker_id is None
    assert graph.states["bad-root"].reason == "User function failed."
    assert graph.states["bad-child"].reason == "Dependency bad-root did not succeed."
    assert list(graph.ready) == ["good-root"]
    assert not graph.complete

    assert _run_next(graph) == "good-root"
    assert _run_next(graph) == "good-child"
    assert graph.complete
    assert not graph.ready
    assert graph.states["join"].state == "failed"
    assert graph.states["good-child"].state == "done"


def test_matching_events_are_idempotent_and_stale_attempts_cannot_advance_graph() -> None:
    graph = ReadyGraph([_work("first"), _work("next", "first")])
    assert graph.ready.popleft() == "first"
    assert not graph.running("first", "attempt-old")
    assert not graph.apply_result("first", "attempt-old", success=True)

    graph.assign("first", "attempt-current", "worker-1")
    assert not graph.running("first", "attempt-old")
    assert not graph.apply_result("first", "attempt-old", success=False)
    assert graph.states["first"].state == "pending"
    assert graph.running("first", "attempt-current")
    graph.running("first", "attempt-current")
    assert graph.states["first"].state == "running"
    assert not graph.ready

    assert graph.apply_result("first", "attempt-current", success=True)
    assert not graph.apply_result("first", "attempt-current", success=False, reason="Duplicate failure.")
    assert not graph.apply_result("first", "attempt-old", success=False)
    assert not graph.running("first", "attempt-current")
    graph.finish("first", success=False, reason="Late scheduler error.")

    assert graph.states["first"].state == "done"
    assert graph.states["first"].reason is None
    assert graph.remaining["next"] == 0
    assert list(graph.ready) == ["next"]
    with pytest.raises(ValueError, match="Only a ready, unassigned work unit"):
        graph.assign("first", "new-attempt", "worker-2")


def test_result_can_arrive_before_started_event_without_replaying_execution() -> None:
    graph = ReadyGraph([_work("task")])
    graph.ready.popleft()
    graph.assign("task", "attempt", "worker")
    assert graph.apply_result("task", "attempt", success=True)
    assert not graph.running("task", "attempt")
    assert graph.complete
    assert graph.states["task"].state == "done"


@pytest.mark.parametrize(
    ("nodes", "message"),
    [
        ([_work("same"), _work("same")], "Duplicate logical job"),
        ([_work("child", "missing")], "Unknown logical dependency"),
        ([_work("child", "parent", "parent"), _work("parent")], "Duplicate dependency"),
        ([_work("self", "self")], "dependency cycle"),
        ([_work("one", "two"), _work("two", "one")], "dependency cycle"),
        ([_work("root"), _work("one", "root", "two"), _work("two", "one")], "dependency cycle"),
    ],
)
def test_malformed_graph_is_rejected_before_any_assignment(nodes: list[GraphWork], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ReadyGraph(nodes)


class _CountedEdges(list[str]):
    """Count edge visits without timing-sensitive performance assertions."""

    def __init__(self, values: list[str]) -> None:
        super().__init__(values)
        self.visits = 0

    def __iter__(self) -> Iterator[str]:
        for value in super().__iter__():
            self.visits += 1
            yield value


@pytest.mark.parametrize("size", [1_000, 10_000])
@pytest.mark.parametrize("shape", ["chain", "fanout"])
def test_large_graphs_visit_each_affected_edge_once_and_ready_queue_is_bounded(size: int, shape: str) -> None:
    if shape == "chain":
        nodes = [_work(f"job-{index}", *([f"job-{index - 1}"] if index else [])) for index in range(size)]
        frontier_limit = 1
    else:
        branches = [f"job-{index}" for index in range(1, size - 1)]
        nodes = [_work("root"), *(_work(key, "root") for key in branches), _work("join", *branches)]
        frontier_limit = size - 2
    edge_count = sum(len(node.dependencies) for node in nodes)
    tracked_dependencies = []
    for node in nodes:
        tracked = _CountedEdges(node.dependencies)
        node.dependencies = tracked
        tracked_dependencies.append(tracked)
    graph = ReadyGraph(reversed(nodes))
    assert sum(edges.visits for edges in tracked_dependencies) <= 3 * edge_count
    tracked_dependents = {key: _CountedEdges(children) for key, children in graph.dependents.items()}
    graph.dependents = dict(tracked_dependents)

    seen: set[str] = set()
    peak_ready = 0
    while graph.ready:
        peak_ready = max(peak_ready, len(graph.ready))
        key = _run_next(graph)
        assert key not in seen
        seen.add(key)
    assert graph.complete
    assert len(seen) == size
    assert peak_ready == frontier_limit
    assert sum(edges.visits for edges in tracked_dependents.values()) == edge_count
    assert not graph.active


def test_deep_failure_propagation_is_iterative_and_does_not_queue_descendants() -> None:
    size = 10_000
    nodes = [_work(f"job-{index}", *([f"job-{index - 1}"] if index else [])) for index in range(size)]
    graph = ReadyGraph(reversed(nodes))
    assert graph.ready.popleft() == "job-0"
    graph.finish("job-0", success=False, reason="Allocation unavailable.")

    assert graph.complete
    assert not graph.ready
    assert not graph.active
    assert all(state.state == "failed" for state in graph.states.values())
    assert all(state.attempt_id is None for state in graph.states.values())


def test_manifest_roundtrip_preserves_resource_requests_and_staged_commands() -> None:
    manifest = RunManifest(
        run_id="run-1",
        snapshot_key="snapshot-1",
        nodes=[_work("root"), _work("child", "root")],
        agents=[AgentWork("worker-1", "cpu", "agent-job-1", ["python", "agent.py"], {}, "agent.log")],
    )

    assert msgspec.json.decode(msgspec.json.encode(manifest), type=RunManifest) == manifest


def test_read_run_state_returns_typed_batched_index(tmp_path: Path) -> None:
    workspace = DiskWorkspace(directory=str(tmp_path))
    expected = RunState(
        run_id="run-1",
        jobs={"job-1": LogicalState("done", None, "attempt-1", "worker-1")},
        status="done",
    )
    workspace.put_job_file("run-1", "run-state.json", msgspec.json.encode(expected))

    assert read_run_state(workspace, "run-1") == expected


@pytest.mark.parametrize(
    "data",
    [
        b"not-json",
        b"[]",
        b'{"run_id":"run-1"}',
        b'{"run_id":"run-1","jobs":{},"version":2}',
        b'{"run_id":"run-1","jobs":{},"unexpected":true}',
        b'{"run_id":"run-1","jobs":{},"status":"unrecognized"}',
        b'{"run_id":"run-1","jobs":{"job-1":{"state":"unrecognized"}}}',
    ],
)
def test_read_run_state_wraps_malformed_records_as_storage_errors(tmp_path: Path, data: bytes) -> None:
    workspace = DiskWorkspace(directory=str(tmp_path))
    workspace.put_job_file("run-1", "run-state.json", data)

    with pytest.raises(StorageError, match="Invalid graph run state") as exc_info:
        read_run_state(workspace, "run-1")
    assert isinstance(exc_info.value.__cause__, msgspec.DecodeError)


def test_read_run_state_rejects_another_runs_index(tmp_path: Path) -> None:
    workspace = DiskWorkspace(directory=str(tmp_path))
    workspace.put_job_file("run-1", "run-state.json", msgspec.json.encode(RunState("run-2", {})))

    with pytest.raises(StorageError, match="different run"):
        read_run_state(workspace, "run-1")


def test_read_run_state_preserves_not_yet_published_state(tmp_path: Path) -> None:
    workspace = DiskWorkspace(directory=str(tmp_path))

    with pytest.raises(FileNotFoundError):
        read_run_state(workspace, "run-1")
