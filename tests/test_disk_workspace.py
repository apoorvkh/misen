"""Unit tests for :class:`misen.workspaces.disk.FileKVMapping`.

The file-backed mapping replaces the old LMDB index with one atomically-renamed
file per key plus close-to-open reads, which is what makes ``.misen`` tolerant
of NFS. These tests pin the round-trip, overwrite, error, iteration, and
concurrency behaviour, and include a cross-"host" simulation (two independent
instances over one directory) as the most direct NFS-safety proof reachable
within a single filesystem.
"""
# ruff: noqa: D103, S101

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest

from misen.utils.hashing import ResolvedTaskHash, ResultHash, TaskHash
from misen.workspaces.disk import FileKVMapping

if TYPE_CHECKING:
    from pathlib import Path


def _resolved_cache(tmp_path: Path) -> FileKVMapping[TaskHash, ResolvedTaskHash]:
    return FileKVMapping[TaskHash, ResolvedTaskHash](tmp_path / "resolved_hash_cache")


def test_round_trip_preserves_value_and_type(tmp_path: Path) -> None:
    cache = _resolved_cache(tmp_path)
    key = TaskHash(0x1234)
    value = ResolvedTaskHash(0x5678)

    cache[key] = value

    fetched = cache[key]
    assert fetched == value
    assert isinstance(fetched, ResolvedTaskHash)
    assert len(cache) == 1
    assert list(cache) == [key]


def test_other_parameterization_round_trips(tmp_path: Path) -> None:
    """The ``resolved -> result`` mapping used by DiskWorkspace behaves identically."""
    cache: FileKVMapping[ResolvedTaskHash, ResultHash] = FileKVMapping[ResolvedTaskHash, ResultHash](
        tmp_path / "result_hash_cache"
    )
    cache[ResolvedTaskHash(0xAA)] = ResultHash(0xBB)
    fetched = cache[ResolvedTaskHash(0xAA)]
    assert fetched == ResultHash(0xBB)
    assert isinstance(fetched, ResultHash)


def test_setitem_overwrites(tmp_path: Path) -> None:
    """Unlike the write-once result store, the index overwrites in place."""
    cache = _resolved_cache(tmp_path)
    key = TaskHash(0xABCD)

    cache[key] = ResolvedTaskHash(0x1111)
    cache[key] = ResolvedTaskHash(0x2222)

    assert cache[key] == ResolvedTaskHash(0x2222)
    assert len(cache) == 1


def test_sharded_layout(tmp_path: Path) -> None:
    cache = _resolved_cache(tmp_path)
    key = TaskHash(0x1234)
    cache[key] = ResolvedTaskHash(0x5678)

    b32 = key.b32()
    expected = tmp_path / "resolved_hash_cache" / b32[:2] / b32
    assert expected.is_file()


def test_getitem_missing_raises_keyerror(tmp_path: Path) -> None:
    cache = _resolved_cache(tmp_path)
    with pytest.raises(KeyError):
        _ = cache[TaskHash(0xDEAD)]


def test_delitem_missing_raises_keyerror(tmp_path: Path) -> None:
    cache = _resolved_cache(tmp_path)
    with pytest.raises(KeyError):
        del cache[TaskHash(0xDEAD)]


def test_delitem_removes_entry(tmp_path: Path) -> None:
    cache = _resolved_cache(tmp_path)
    key = TaskHash(0x42)
    cache[key] = ResolvedTaskHash(0x99)

    del cache[key]

    assert key not in cache
    assert len(cache) == 0
    with pytest.raises(KeyError):
        _ = cache[key]


def test_contains_true_false_and_wrong_type(tmp_path: Path) -> None:
    cache = _resolved_cache(tmp_path)
    key = TaskHash(0x1)
    cache[key] = ResolvedTaskHash(0x2)

    assert key in cache
    assert TaskHash(0x999) not in cache
    # A key of the wrong hash type is never "in" the mapping, even if the
    # underlying integer matches a stored key.
    assert ResolvedTaskHash(0x1) not in cache
    assert "not-a-hash" not in cache


def test_iter_and_len_ignore_tmp_and_foreign_files(tmp_path: Path) -> None:
    cache = _resolved_cache(tmp_path)
    keys = {TaskHash(0x1), TaskHash(0x2222), TaskHash(0xABCDEF)}
    for k in keys:
        cache[k] = ResolvedTaskHash(int(k) * 2)

    root = tmp_path / "resolved_hash_cache"
    # A leftover temp file (leading dot) must be ignored: the [A-Z2-7] glob
    # charset structurally excludes it.
    sample = next(iter(keys)).b32()
    (root / sample[:2] / f".{sample}.deadbeef.tmp").write_bytes(b"\x00" * 8)
    # Foreign / undecodable names in a valid-looking shard must be ignored:
    # wrong length, lowercase (outside the base32 alphabet), and a file whose
    # name does not match its shard prefix.
    (root / sample[:2] / "TOOSHORT").write_bytes(b"\x00" * 8)
    (root / sample[:2] / "lowercasenames").write_bytes(b"\x00" * 8)
    foreign_shard = root / "ZZ"
    foreign_shard.mkdir(parents=True, exist_ok=True)
    (foreign_shard / sample).write_bytes(b"\x00" * 8)  # name[:2] != "ZZ"

    assert set(cache) == keys
    assert len(cache) == len(keys)


def test_concurrent_same_key_writers_are_safe(tmp_path: Path) -> None:
    """Deterministic values mean concurrent writers race harmlessly to the same bytes."""
    cache = _resolved_cache(tmp_path)
    key = TaskHash(0x7777)
    value = ResolvedTaskHash(0x8888)

    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            barrier.wait()
            for _ in range(50):
                cache[key] = value
        except BaseException as e:  # noqa: BLE001  -- surface any thread failure to the assertion
            errors.append(e)

    threads = [threading.Thread(target=writer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert cache[key] == value
    assert len(cache) == 1


def test_cross_host_simulation_two_instances_one_directory(tmp_path: Path) -> None:
    """Two instances over one dir model two hosts: writes/deletes via one are seen by the other.

    Close-to-open reads (every access re-opens the file) are exactly what makes
    this hold on NFS; here it is proven within a single filesystem.
    """
    directory = tmp_path / "resolved_hash_cache"
    host_a: FileKVMapping[TaskHash, ResolvedTaskHash] = FileKVMapping[TaskHash, ResolvedTaskHash](directory)
    host_b: FileKVMapping[TaskHash, ResolvedTaskHash] = FileKVMapping[TaskHash, ResolvedTaskHash](directory)

    key = TaskHash(0x5151)

    # Write via A, read via B.
    host_a[key] = ResolvedTaskHash(0x1)
    assert host_b[key] == ResolvedTaskHash(0x1)

    # Overwrite via A, B sees the new value (no stale mmap).
    host_a[key] = ResolvedTaskHash(0x2)
    assert host_b[key] == ResolvedTaskHash(0x2)

    # Delete via B, A sees it as absent.
    del host_b[key]
    assert key not in host_a
    with pytest.raises(KeyError):
        _ = host_a[key]


def test_construct_without_parameterization_raises(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="FileKVMapping"):
        FileKVMapping(tmp_path / "unparam")
