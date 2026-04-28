"""Tests for :class:`misen.workspaces.memory.InMemoryWorkspace`."""
# ruff: noqa: D103, S101, SLF001, PLR2004

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from misen import Task, meta
from misen.exceptions import LockUnavailableError
from misen.executors.in_process import InProcessExecutor
from misen.utils.hashing import ResolvedTaskHash, ResultHash
from misen.utils.settings import ConfigurableMeta
from misen.workspace import Workspace
from misen.workspaces.memory import InMemoryWorkspace

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clear_singleton_cache() -> None:
    # InMemoryWorkspace() with the same kwargs is memoized by Configurable;
    # tests need fresh instances so they don't share state through the cache.
    ConfigurableMeta._instances.clear()


_call_log: list[int] = []


@meta(id="memory_ws_doubler", cache=True)
def _doubler(x: int) -> int:
    _call_log.append(x)
    return x * 2


@meta(id="memory_ws_chain_a", cache=True)
def _chain_a() -> int:
    return 7


@meta(id="memory_ws_chain_b", cache=True)
def _chain_b(value: int) -> int:
    return value + 1


def test_resolve_type_memory_alias() -> None:
    assert Workspace.resolve_type("memory") is InMemoryWorkspace


def test_default_directory_is_a_fresh_tempdir() -> None:
    ws = InMemoryWorkspace()
    try:
        assert ws._directory.exists()
        assert ws._owns_directory is True
        assert ws.get_temp_dir().exists()
    finally:
        ws.close()


def test_close_removes_owned_tempdir() -> None:
    ws = InMemoryWorkspace()
    tempdir = ws._directory
    assert tempdir.exists()
    ws.close()
    assert not tempdir.exists()


def test_close_leaves_explicit_directory_intact(tmp_path: Path) -> None:
    target = tmp_path / "kept"
    ws = InMemoryWorkspace(directory=str(target))
    assert ws._owns_directory is False
    ws.close()
    assert target.exists()


def test_hash_round_trip(tmp_path: Path) -> None:
    ws = InMemoryWorkspace(directory=str(tmp_path / "ws"))

    task = Task(_doubler, x=3)
    resolved = ResolvedTaskHash(0xDEAD_BEEF_1234_5678)
    result = ResultHash(0x0011_2233_4455_6677)

    assert ws.get_resolved_hash(task) is None
    ws.set_resolved_hash(task, resolved)
    assert ws.get_resolved_hash(task) == resolved

    ws.set_result_hash(task, result)
    assert ws.get_result_hash(task) == result


def test_lock_per_key_is_independent(tmp_path: Path) -> None:
    ws = InMemoryWorkspace(directory=str(tmp_path / "ws"))

    a = ws.lock(namespace="task", key="alpha")
    b = ws.lock(namespace="task", key="beta")

    with a.context():
        # Different key acquires immediately even while ``a`` is held.
        b.acquire(blocking=False)
        b.release()


def test_lock_same_key_returns_same_object(tmp_path: Path) -> None:
    ws = InMemoryWorkspace(directory=str(tmp_path / "ws"))
    first = ws.lock(namespace="task", key="alpha")
    second = ws.lock(namespace="task", key="alpha")
    assert first is second


def test_lock_contention_raises_on_non_blocking(tmp_path: Path) -> None:
    ws = InMemoryWorkspace(directory=str(tmp_path / "ws"))

    held = ws.lock(namespace="task", key="busy")
    contender = ws.lock(namespace="task", key="busy")
    with held.context():
        with pytest.raises(LockUnavailableError):
            contender.acquire(blocking=False)


def test_in_process_executor_caches_result(tmp_path: Path) -> None:
    ws = InMemoryWorkspace(directory=str(tmp_path / "ws"))
    executor = InProcessExecutor()
    _call_log.clear()

    task = Task(_doubler, x=21)
    executor.submit(tasks={task}, workspace=ws, blocking=True)
    assert ws.results[task] == 42
    assert _call_log == [21]

    # Second submit hits cache; the function does not run again.
    executor.submit(tasks={task}, workspace=ws, blocking=True)
    assert _call_log == [21]
    assert ws.results[task] == 42


def test_in_process_executor_runs_dependency_chain(tmp_path: Path) -> None:
    ws = InMemoryWorkspace(directory=str(tmp_path / "ws"))
    executor = InProcessExecutor()

    a_task = Task(_chain_a)
    b_task = Task(_chain_b, value=a_task.T)
    executor.submit(tasks={b_task}, workspace=ws, blocking=True)

    assert ws.results[a_task] == 7
    assert ws.results[b_task] == 8
