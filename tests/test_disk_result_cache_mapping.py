from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import pytest

from misen.task import ResultHash
from misen.workspaces.disk import DiskResultCacheMapping

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def cache(tmp_path: Path) -> DiskResultCacheMapping:
    d = tmp_path / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return DiskResultCacheMapping(d)


def test_get_missing_raises_keyerror(cache: DiskResultCacheMapping) -> None:
    with pytest.raises(KeyError):
        _ = cache[ResultHash(123)]


def test_contains_and_len_empty(cache: DiskResultCacheMapping) -> None:
    assert ResultHash(1) not in cache
    assert len(cache) == 0
    assert list(cache) == []


def test_set_then_get_and_contains(cache: DiskResultCacheMapping) -> None:
    k = ResultHash(0xABC)
    v = b"payload"

    cache[k] = v
    assert k in cache
    assert cache[k] == v
    assert len(cache) == 1


def test_write_once_does_not_overwrite_existing_value(cache: DiskResultCacheMapping) -> None:
    k = ResultHash(42)
    v1 = b"first"
    v2 = b"SECOND_DIFFERENT"

    cache[k] = v1
    cache[k] = v2  # should be ignored (write-once)

    assert cache[k] == v1
    assert len(cache) == 1


def test_delete_then_can_write_again(cache: DiskResultCacheMapping) -> None:
    k = ResultHash(7)

    cache[k] = b"a"
    assert cache[k] == b"a"

    del cache[k]
    assert k not in cache
    with pytest.raises(KeyError):
        _ = cache[k]

    # after delete, a new fill should work
    cache[k] = b"b"
    assert cache[k] == b"b"
    assert len(cache) == 1


def test_del_missing_raises_keyerror(cache: DiskResultCacheMapping) -> None:
    with pytest.raises(KeyError):
        del cache[ResultHash(999)]


def test_iter_yields_only_valid_sharded_paths(cache: DiskResultCacheMapping, tmp_path: Path) -> None:
    # Two valid entries
    k1 = ResultHash(0x00000000000000AA)
    k2 = ResultHash(0x00000000000000BB)
    cache[k1] = b"a"
    cache[k2] = b"b"

    keys = list(cache)
    assert set(keys) == {k1, k2}

    # Create a bogus file in the right shape but wrong shard prefix (should be ignored)
    # e.g. key starts with "aa" but put it under "bb/"
    bad_key_hex = k1.hex()
    wrong_dir = cache.directory / "bb"
    wrong_dir.mkdir(parents=True, exist_ok=True)
    (wrong_dir / f"{bad_key_hex}.dill").write_bytes(b"bogus")

    keys2 = list(cache)
    assert set(keys2) == {k1, k2}


def test_dict_items_keys_values_views(cache: DiskResultCacheMapping) -> None:
    k1, k2 = ResultHash(1), ResultHash(2)
    cache[k1] = b"a"
    cache[k2] = b"b"

    assert dict(cache) == {k1: b"a", k2: b"b"}
    assert set(cache.keys()) == {k1, k2}
    assert set(cache.values()) == {b"a", b"b"}
    assert set(cache.items()) == {(k1, b"a"), (k2, b"b")}


def test_update_is_fill_only(cache: DiskResultCacheMapping) -> None:
    k = ResultHash(5)
    cache[k] = b"orig"

    cache.update({k: b"new"})  # update uses __setitem__ => should not overwrite
    assert cache[k] == b"orig"


def test_persistence_across_interpreter_sessions(tmp_path: Path) -> None:
    # This uses two *separate* python processes.
    cache_dir = tmp_path / "cache_dir"
    cache_dir.mkdir(parents=True, exist_ok=True)

    writer = textwrap.dedent(
        r"""
        from pathlib import Path
        import sys

        from misen.task import Hash as ResultHash
        from misen.workspaces.disk import DiskResultCacheMapping

        d = Path(sys.argv[1])
        c = DiskResultCacheMapping(d)

        c[ResultHash(1)] = b"one"
        c[ResultHash(2)] = b"two"
        c[ResultHash(2)] = b"TWO_OVERWRITE_ATTEMPT"  # should be ignored
        assert c[ResultHash(2)] == b"two"
        """
    )

    reader = textwrap.dedent(
        r"""
        from pathlib import Path
        import sys

        from misen.task import ResultHash
        from misen.workspaces.disk import DiskResultCacheMapping

        d = Path(sys.argv[1])
        c = DiskResultCacheMapping(d)

        assert ResultHash(1) in c
        assert ResultHash(2) in c
        assert c[ResultHash(1)] == b"one"
        assert c[ResultHash(2)] == b"two"
        assert set(c) == {ResultHash(1), ResultHash(2)}
        """
    )

    subprocess.run([sys.executable, "-c", writer, str(cache_dir)], check=True)
    subprocess.run([sys.executable, "-c", reader, str(cache_dir)], check=True)


def test_two_processes_competing_write_once(tmp_path: Path) -> None:
    # Competing writers should serialize via the per-key lock, and the value should
    # end up as exactly one of the full payloads (never partial, never mixed).
    cache_dir = tmp_path / "cache_dir"
    cache_dir.mkdir(parents=True, exist_ok=True)

    worker = textwrap.dedent(
        r"""
        from pathlib import Path
        import sys
        import os

        from misen.task import Hash as ResultHash
        from misen.workspaces.disk import DiskResultCacheMapping

        d = Path(sys.argv[1])
        payload = sys.argv[2].encode("utf-8")
        c = DiskResultCacheMapping(d)

        k = ResultHash(123)
        c[k] = payload
        """
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    p1 = subprocess.Popen([sys.executable, "-c", worker, str(cache_dir), "FIRST"], env=env)
    p2 = subprocess.Popen([sys.executable, "-c", worker, str(cache_dir), "SECOND"], env=env)
    assert p1.wait() == 0
    assert p2.wait() == 0

    c = DiskResultCacheMapping(cache_dir)
    v = c[ResultHash(123)]
    assert v in (b"FIRST", b"SECOND")
