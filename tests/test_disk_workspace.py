"""Regression tests for :class:`misen.workspaces.disk.LMDBMapping`.

Each mutation path (``__setitem__``, ``__delitem__``, ``clear``) ends with a
forced ``env.sync()`` so writes are durable before the NFS lock is released.
``lmdb.Environment.sync`` is a C method that rejects keyword arguments -- a
prior version passed ``force=True`` and raised ``TypeError`` at every write
site.  These tests pin the call shape by exercising each path end-to-end.
"""
# ruff: noqa: D103, S101, PLR2004

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from misen.utils.hashing import ResolvedTaskHash, ResultHash, TaskHash
from misen.workspaces.disk import LMDBMapping

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def mapping(tmp_path: Path) -> Iterator[LMDBMapping[TaskHash, ResolvedTaskHash]]:
    m: LMDBMapping[TaskHash, ResolvedTaskHash] = LMDBMapping[TaskHash, ResolvedTaskHash](
        tmp_path / "test.mdb",
    )
    yield m
    m.close()


def test_setitem_triggers_sync_without_typeerror(
    mapping: LMDBMapping[TaskHash, ResolvedTaskHash],
) -> None:
    """``__setitem__`` writes and forces an env.sync -- must not raise."""
    key = TaskHash(0x1234)
    value = ResolvedTaskHash(0x5678)
    mapping[key] = value
    assert mapping[key] == value
    assert len(mapping) == 1


def test_delitem_triggers_sync_without_typeerror(
    mapping: LMDBMapping[TaskHash, ResolvedTaskHash],
) -> None:
    """``__delitem__`` removes and forces an env.sync -- must not raise."""
    key = TaskHash(0xABCD)
    mapping[key] = ResolvedTaskHash(0xEF01)
    del mapping[key]
    assert key not in mapping
    assert len(mapping) == 0


def test_clear_triggers_sync_without_typeerror(
    mapping: LMDBMapping[TaskHash, ResolvedTaskHash],
) -> None:
    """``clear`` removes all entries and forces an env.sync -- must not raise."""
    for i in range(5):
        mapping[TaskHash(i)] = ResolvedTaskHash(i * 10)
    assert len(mapping) == 5
    mapping.clear()
    assert len(mapping) == 0


def test_resolved_hash_to_result_hash_mapping(tmp_path: Path) -> None:
    """Spot-check the other concrete parameterization used by DiskWorkspace."""
    m: LMDBMapping[ResolvedTaskHash, ResultHash] = LMDBMapping[ResolvedTaskHash, ResultHash](
        tmp_path / "results.mdb",
    )
    try:
        m[ResolvedTaskHash(0xAA)] = ResultHash(0xBB)
        assert m[ResolvedTaskHash(0xAA)] == ResultHash(0xBB)
    finally:
        m.close()
