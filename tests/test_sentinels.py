"""Tests for runtime-sentinel validation, auto-exclusion, and resolution."""
# ruff: noqa: D103, PLR2004, S101

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from misen import SCRATCH_DIR, Task, meta
from misen.workspaces.memory import InMemoryWorkspace

if TYPE_CHECKING:
    from pathlib import Path

_cached_calls: list[int] = []


@meta(id="sentinel_with_default", cache=False)
def _with_default(scratch_dir: Path = SCRATCH_DIR) -> str:
    return type(scratch_dir).__name__


@meta(id="sentinel_with_default_excluded", cache=False, exclude={"scratch_dir"})
def _with_default_excluded(scratch_dir: Path = SCRATCH_DIR) -> str:
    return type(scratch_dir).__name__


@meta(id="sentinel_plain", cache=False)
def _plain(scratch_dir: Path) -> str:
    return type(scratch_dir).__name__


@meta(id="sentinel_nested", cache=False)
def _nested(dirs: list[Path]) -> str:
    return str(dirs)


@meta(id="sentinel_nested_mapping", cache=False)
def _nested_mapping(dirs: dict[str, Path]) -> str:
    return str(dirs)


@meta(id="sentinel_cached", cache=True)
def _cached(scratch_dir: Path, value: int) -> int:
    assert scratch_dir.is_dir()
    _cached_calls.append(value)
    return value * 2


# Same id with and without a manual exclude: dropping the boilerplate must not
# change task identity (existing workspace caches stay valid).
@meta(id="sentinel_hash_stability", cache=False, exclude={"scratch_dir"})
def _hash_manual_exclude(scratch_dir: Path, value: int) -> int:
    _ = scratch_dir
    return value


@meta(id="sentinel_hash_stability", cache=False)
def _hash_auto_exclude(scratch_dir: Path, value: int) -> int:
    _ = scratch_dir
    return value


def test_signature_default_rejected_at_construction() -> None:
    with pytest.raises(TypeError, match="function-signature default") as excinfo:
        Task(_with_default)
    message = str(excinfo.value)
    assert "Task(_with_default, scratch_dir=SCRATCH_DIR)" in message


def test_signature_default_rejected_even_with_manual_exclude() -> None:
    # The reporter's configuration: default + exclude used to leak the raw
    # sentinel into the function body at runtime. Must now fail at graph build.
    with pytest.raises(TypeError, match="function-signature default"):
        Task(_with_default_excluded)


def test_signature_default_allowed_when_explicitly_bound(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "ws"))
    task = Task(_with_default, scratch_dir=SCRATCH_DIR)
    assert task.result(workspace=workspace) == "PosixPath"


def test_nested_sentinel_rejected_in_kwarg() -> None:
    with pytest.raises(TypeError, match="top-level") as excinfo:
        Task(_nested, dirs=[SCRATCH_DIR])
    assert "argument 'dirs'" in str(excinfo.value)


def test_nested_sentinel_rejected_in_positional_arg() -> None:
    with pytest.raises(TypeError, match="top-level") as excinfo:
        Task(_nested, [SCRATCH_DIR])
    assert "positional argument 0" in str(excinfo.value)


def test_nested_sentinel_rejected_in_dict_value() -> None:
    with pytest.raises(TypeError, match="top-level"):
        Task(_nested_mapping, dirs={"scratch": SCRATCH_DIR})


def test_no_manual_exclude_needed(tmp_path: Path) -> None:
    # Previously raised "Resolved task arguments cannot contain sentinel
    # values" unless @meta(exclude={"scratch_dir"}) was set.
    workspace = InMemoryWorkspace(directory=str(tmp_path / "ws"))
    task = Task(_plain, scratch_dir=SCRATCH_DIR)
    assert task.result(workspace=workspace) == "PosixPath"


def test_manual_exclude_hash_unchanged() -> None:
    manual = Task(_hash_manual_exclude, scratch_dir=SCRATCH_DIR, value=1)
    auto = Task(_hash_auto_exclude, scratch_dir=SCRATCH_DIR, value=1)
    assert manual.task_hash() == auto.task_hash()


def test_cacheable_task_resolves_and_caches(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(directory=str(tmp_path / "ws"))
    _cached_calls.clear()

    task = Task(_cached, scratch_dir=SCRATCH_DIR, value=21)
    assert task.result(workspace=workspace, compute_if_uncached=True) == 42
    assert _cached_calls == [21]

    # Second resolution is a cache hit; the function does not run again.
    assert task.result(workspace=workspace) == 42
    assert _cached_calls == [21]
