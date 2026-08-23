"""Tests for allocation-scoped DASK_CLIENT realization."""
# ruff: noqa: D103, S101

from __future__ import annotations

from typing import Any, ClassVar, cast

import distributed
import pytest

from misen import DASK_CLIENT, Task, meta
from misen.exceptions import ExecutionError
from misen.utils.runtime_values import RuntimeValues
from misen.utils.work_unit import WorkUnit
from misen.workspaces.memory import InMemoryWorkspace

_seen_clients: list[Any] = []


@meta(id="dask_runtime_leaf", cache=False, resources={"nodes": 2})
def _client_leaf(client: Any) -> int:
    _seen_clients.append(client)
    return id(client)


@meta(id="dask_runtime_root", cache=True, resources={"nodes": 2})
def _client_root(leaf_client_id: int, client: Any) -> bool:
    _seen_clients.append(client)
    return leaf_client_id == id(client)


@meta(id="dask_runtime_three_node_root", cache=True, resources={"nodes": 3})
def _three_node_root(value: None) -> None:
    _ = value


class _EqualityHostile:
    """Value whose equality operator must not run during sentinel detection."""

    __hash__ = object.__hash__

    def __eq__(self, _other: object) -> bool:
        raise AssertionError


@meta(id="dask_runtime_equality_hostile", cache=True, exclude={"value"})
def _equality_hostile(value: Any) -> None:
    _ = value


class _FakeClient:
    """Minimal synchronous distributed.Client used to observe ownership."""

    created: ClassVar[list[_FakeClient]] = []
    topologies: ClassVar[list[dict[str, dict[str, str]]]] = []

    def __init__(self, *_: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.shutdown_called = False
        self.waited_for: tuple[int, float] | None = None
        self._scheduler_info_calls = 0
        type(self).created.append(self)

    def wait_for_workers(self, workers: int, *, timeout: float) -> None:
        self.waited_for = (workers, timeout)

    def scheduler_info(self) -> dict[str, Any]:
        index = min(self._scheduler_info_calls, len(type(self).topologies) - 1)
        self._scheduler_info_calls += 1
        return {"workers": type(self).topologies[index]}

    def close(self) -> None:
        self.closed = True

    def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    _FakeClient.created.clear()
    _FakeClient.topologies = [
        {
            "tcp://worker-0": {},
            "tcp://worker-1": {},
        }
    ]
    monkeypatch.setattr(distributed, "Client", _FakeClient)
    monkeypatch.setenv("MISEN_DASK_SCHEDULER_ADDRESS", "tcp://scheduler:8786")
    monkeypatch.setenv("MISEN_DASK_EXPECTED_WORKERS", "2")
    return _FakeClient


def test_work_unit_lazily_reuses_one_client(fake_client: type[_FakeClient], tmp_path: Any) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "ws"))
    leaf = Task(_client_leaf, DASK_CLIENT)
    work_unit = WorkUnit(root=Task(_client_root, cast("int", leaf), DASK_CLIENT), dependencies=set())
    _seen_clients.clear()

    WorkUnit.execute(
        work_unit.graph,
        workspace=workspace,
        job_id="dask-runtime",
    )

    assert work_unit.uses_dask_client
    assert len(fake_client.created) == 1
    assert _seen_clients == [fake_client.created[0], fake_client.created[0]]
    assert fake_client.created[0].waited_for == (2, 600)
    assert fake_client.created[0].kwargs == {"set_as_default": False, "timeout": 600}
    assert fake_client.created[0].closed
    assert not fake_client.created[0].shutdown_called


def test_cached_work_unit_does_not_resolve_client(
    fake_client: type[_FakeClient], monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "ws"))
    work_unit = WorkUnit(root=Task(_client_root, 0, DASK_CLIENT), dependencies=set())
    WorkUnit.execute(
        work_unit.graph,
        workspace=workspace,
        job_id="seed-cache",
    )
    fake_client.created.clear()
    monkeypatch.delenv("MISEN_DASK_SCHEDULER_ADDRESS", raising=False)

    # The default environment-backed spec is invalid on this process, but a
    # cache hit returns before sentinel resolution and therefore never reads it.
    WorkUnit.execute(work_unit.graph, workspace=workspace, job_id="cache-hit")
    assert fake_client.created == []


def test_runtime_membership_change_fails_and_closes(fake_client: type[_FakeClient]) -> None:
    fake_client.topologies = [
        {
            "tcp://worker-0": {},
            "tcp://worker-1": {},
        },
        {
            "tcp://worker-0": {},
            "tcp://worker-2": {},
        },
    ]

    with pytest.raises(ExecutionError, match="membership changed"):
        with RuntimeValues() as values:
            values.resolve(DASK_CLIENT)

    assert fake_client.created[0].closed
    assert not fake_client.created[0].shutdown_called


def test_runtime_rejects_a_single_worker_group(fake_client: type[_FakeClient], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISEN_DASK_EXPECTED_WORKERS", "1")

    with pytest.raises(ExecutionError, match="must be at least 2"):
        with RuntimeValues() as values:
            values.resolve(DASK_CLIENT)

    assert fake_client.created == []


def test_client_closes_when_work_unit_fails(fake_client: type[_FakeClient]) -> None:
    def fail() -> None:
        with RuntimeValues() as values:
            values.resolve(DASK_CLIENT)
            msg = "task failed"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="task failed"):
        fail()

    assert fake_client.created[0].closed


def test_dask_setup_preserves_unexpected_errors(
    fake_client: type[_FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_wait(self: _FakeClient, workers: int, *, timeout: float) -> None:
        _ = self, workers, timeout
        raise AssertionError("client bug")

    monkeypatch.setattr(_FakeClient, "wait_for_workers", fail_wait)

    with pytest.raises(AssertionError, match="client bug"):
        with RuntimeValues() as values:
            values.resolve(DASK_CLIENT)

    assert fake_client.created[0].closed


def test_dask_close_interrupt_is_not_swallowed(fake_client: type[_FakeClient], monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupt_close(self: _FakeClient) -> None:
        self.closed = True
        raise KeyboardInterrupt

    monkeypatch.setattr(_FakeClient, "close", interrupt_close)

    with pytest.raises(KeyboardInterrupt):
        with RuntimeValues() as values:
            values.resolve(DASK_CLIENT)

    assert fake_client.created[0].closed


def test_direct_task_execution_cannot_resolve_dask_client(tmp_path: Any) -> None:
    task = Task(_client_leaf, DASK_CLIENT)
    workspace = InMemoryWorkspace(directory=str(tmp_path / "ws"))
    with pytest.raises(ExecutionError, match="WorkUnit through an Executor"):
        task.result(workspace=workspace, compute_if_uncached=True)


def test_dask_task_topology_must_match_work_unit_allocation() -> None:
    child = Task(_client_leaf, DASK_CLIENT)
    with pytest.raises(ValueError, match=r"exact.*topology"):
        WorkUnit(root=Task(_three_node_root, cast("None", child)), dependencies=set())


def test_dask_client_requires_multiple_nodes() -> None:
    with pytest.raises(ValueError, match="DASK_CLIENT requires nodes > 1"):
        WorkUnit(root=Task(_client_leaf, DASK_CLIENT).with_resources(nodes=1), dependencies=set())


def test_resources_can_enable_dask_after_task_construction() -> None:
    work_unit = WorkUnit(
        root=Task(_three_node_root, DASK_CLIENT).with_resources(nodes=2),
        dependencies=set(),
    )

    assert work_unit.uses_dask_client


def test_runtime_sentinel_detection_uses_identity() -> None:
    work_unit = WorkUnit(root=Task(_equality_hostile, _EqualityHostile()), dependencies=set())

    assert not work_unit.uses_dask_client


def test_accelerator_memory_is_a_minimum_not_group_topology() -> None:
    expected_memory = 80
    child = Task(_client_leaf, DASK_CLIENT).with_resources(accelerators=1, accelerator_memory=40)
    root = Task(_client_root, cast("int", child), DASK_CLIENT).with_resources(
        accelerators=1,
        accelerator_memory=expected_memory,
    )
    work_unit = WorkUnit(root=root, dependencies=set())

    assert work_unit.resources["accelerator_memory"] == expected_memory
