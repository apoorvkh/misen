"""Tests for NFSLock and its filesystem-clock-offset machinery."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from misen.exceptions import LockUnavailableError
from misen.utils import locks
from misen.utils.locks import (
    NFSLock,
    _ClockOffsetLock,
    _get_clock_offset,
    _measure_clock_offset,
)


@pytest.fixture(autouse=True)
def _clear_offset_cache() -> None:
    with locks._clock_offset_cache_lock:
        locks._clock_offset_cache.clear()


def test_measure_clock_offset_local_is_small(tmp_path: Path) -> None:
    # On a local filesystem the local clock IS the filesystem's clock, so
    # offset must be far below lock lifetimes (10s+). Allow 2s slack for
    # coarse mtime precision on the probed filesystem.
    offset = _measure_clock_offset(tmp_path)
    assert abs(offset) < timedelta(seconds=2)


def test_get_clock_offset_caches_by_device(tmp_path: Path) -> None:
    stub = timedelta(seconds=5)
    with patch.object(locks, "_measure_clock_offset", return_value=stub) as mock:
        first = _get_clock_offset(tmp_path / "a.lock")
        second = _get_clock_offset(tmp_path / "b.lock")
    assert first == stub
    assert second == stub
    mock.assert_called_once()


def test_get_clock_offset_falls_back_on_stat_error(tmp_path: Path) -> None:
    missing_parent = tmp_path / "no" / "such" / "dir" / "x.lock"
    assert _get_clock_offset(missing_parent) == timedelta(0)


def test_get_clock_offset_falls_back_on_measurement_error(tmp_path: Path) -> None:
    with patch.object(locks, "_measure_clock_offset", side_effect=OSError("probe failed")):
        assert _get_clock_offset(tmp_path / "x.lock") == timedelta(0)


def test_clock_offset_lock_now_applies_offset(tmp_path: Path) -> None:
    lock = _ClockOffsetLock(lockfile=str(tmp_path / "t.lock"), lifetime=10)
    lock._clock_offset = timedelta(seconds=42)

    before = datetime.now()  # noqa: DTZ005
    shifted = lock._now()
    after = datetime.now()  # noqa: DTZ005

    assert before + timedelta(seconds=41) < shifted < after + timedelta(seconds=43)


def test_measurement_is_deferred_until_first_use(tmp_path: Path) -> None:
    with patch.object(locks, "_measure_clock_offset", return_value=timedelta(0)) as mock:
        lock = NFSLock(tmp_path / "lazy.lock", lifetime=5)
        mock.assert_not_called()
        with lock.context():
            pass
        mock.assert_called_once()


def test_second_lock_on_same_fs_does_not_remeasure(tmp_path: Path) -> None:
    with patch.object(locks, "_measure_clock_offset", return_value=timedelta(seconds=3)) as mock:
        with NFSLock(tmp_path / "first.lock", lifetime=5).context():
            pass
        with NFSLock(tmp_path / "second.lock", lifetime=5).context():
            pass
    mock.assert_called_once()


def test_nfs_lock_acquire_release(tmp_path: Path) -> None:
    lock = NFSLock(tmp_path / "e2e.lock", lifetime=5)
    assert not lock.is_locked()
    with lock.context():
        assert lock.is_locked()
    assert not lock.is_locked()


def test_nfs_lock_timeout_surfaces_as_lock_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "busy.lock"
    holder = NFSLock(path, lifetime=30)
    contender = NFSLock(path, lifetime=30)
    with holder.context():
        with pytest.raises(LockUnavailableError):
            contender.acquire(timeout=0)
