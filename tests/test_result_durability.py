"""Crash-safety tests for the result-commit ordering on :class:`DiskWorkspace`.

Durability invariant: a ``resolved_hash -> result_hash`` mapping (the durable
pointer) may exist only if its payload is durably present. ``save_task_result``
enforces this by committing the payload (atomic ``os.rename`` + parent fsync)
*before* writing the pointer, so a ``scancel`` / SIGKILL at any instant leaves
either (no payload, no pointer) or (orphan payload, no pointer) -- both recover
by recomputing -- but never the dangling (pointer, no payload) state that makes
a dependent job hard-fail. Readers must treat a dangling mapping (should one
arise from external payload loss) as "not done" and recompute, never as fatal.
"""
# ruff: noqa: S101, PLR2004, SLF001

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Any

from misen import Task, meta
from misen.utils.hashing import ResultHash
from misen.utils.task_utils import save_task_result
from misen.workspaces import disk as disk_mod
from misen.workspaces.disk import DiskResultStore, DiskWorkspace

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


@meta(id="durability_produce", cache=True)
def _produce(x: int) -> int:
    return x + 1


@meta(id="durability_consume", cache=True)
def _consume(v: int) -> int:
    return v * 10


def _raise_setitem(self: Any, key: Any, value: Any) -> None:  # noqa: ARG001  -- matches __setitem__ shape
    """Stand-in for the durable pointer write that dies mid-commit."""
    msg = "simulated crash: interrupted before the pointer was durably written"
    raise RuntimeError(msg)


def test_payload_committed_before_pointer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A crash at the pointer write leaves an orphan payload, never a dangling pointer."""
    ws = DiskWorkspace(directory=str(tmp_path / "ws"))
    task = Task(_produce, x=1)
    expected = 2
    result_hash = ResultHash.from_object(expected)
    resolved_hash = task.resolved_hash(workspace=ws)

    # Inject a failure into the durable pointer write only (reads still work),
    # standing in for a process kill between the payload and pointer commits.
    monkeypatch.setattr(type(ws._result_hash_cache), "__setitem__", _raise_setitem)

    try:
        save_task_result(task, expected, ws)
    except RuntimeError:
        pass
    else:
        msg = "expected the injected pointer-write failure to propagate"
        raise AssertionError(msg)

    # Payload was committed first -> durably present.
    assert result_hash in ws.results.result_store
    # Pointer was never durably written -> the forbidden (pointer, no payload)
    # state does not exist; this is just a recoverable orphan payload.
    assert resolved_hash not in ws._result_hash_cache

    # A fresh reader (post-crash: process-local hot caches are gone) sees the
    # task as not done and recomputes, rather than trusting a half-written commit.
    ws._result_hashes.clear()
    ws._resolved_hashes.clear()
    assert task.done(workspace=ws) is False


def test_reader_recomputes_on_dangling_mapping(tmp_path: Path) -> None:
    """If a payload is lost out-of-band, every reader treats the task as not done."""
    ws = DiskWorkspace(directory=str(tmp_path / "ws"))
    task = Task(_produce, x=3)

    assert task.result(workspace=ws, compute_if_uncached=True) == 4
    resolved_hash = task.resolved_hash(workspace=ws)
    result_hash = ws.get_result_hash(task)

    # Manufacture the forbidden state directly: keep the pointer, drop the payload.
    shutil.rmtree(ws.results.result_store[result_hash])
    ws._result_hashes.clear()
    ws._resolved_hashes.clear()
    assert resolved_hash in ws._result_hash_cache  # pointer present
    assert result_hash not in ws.results.result_store  # payload absent

    # Doneness readers must report "not done" -- never raise on the missing payload.
    assert (task in ws.results) is False
    assert task.is_cached(workspace=ws) is False
    assert task.done(workspace=ws) is False

    # And the value reader recomputes (and re-materializes the payload) cleanly.
    assert task.result(workspace=ws, compute_if_uncached=True) == 4
    assert result_hash in ws.results.result_store


def test_disk_result_store_publishes_atomically(tmp_path: Path) -> None:
    """``DiskResultStore.__setitem__`` renames the temp dir into place (no copy, no partial)."""
    store = DiskResultStore(tmp_path / "results")
    key = ResultHash(0xABCDEF)

    src = tmp_path / "tmp_payload"
    src.mkdir()
    (src / "data.bin").write_bytes(b"hello")

    store[key] = src
    assert key in store
    assert not src.exists()  # temp dir consumed by the atomic rename
    assert (store[key] / "data.bin").read_bytes() == b"hello"

    # Write-once: storing again while present is a no-op and leaves the payload intact.
    src2 = tmp_path / "tmp_payload2"
    src2.mkdir()
    (src2 / "data.bin").write_bytes(b"overwrite-me")
    store[key] = src2
    assert (store[key] / "data.bin").read_bytes() == b"hello"


def test_disk_result_store_fsyncs_payload_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every payload file is fsync'd before the publish rename, not just the dir entry.

    Serializer backends only close-flush their files, so without an explicit
    contents fsync a crash could publish a directory whose files are empty or
    partial on NFS. Verify each regular file in the payload tree is fsync'd and
    that all those fsyncs precede the atomic ``os.rename`` that publishes it.
    """
    store = DiskResultStore(tmp_path / "results")
    key = ResultHash(0x123456)

    # A nested payload shaped like serde output (manifest + leaves subtree).
    src = tmp_path / "tmp_payload"
    (src / "leaves" / "ndarray").mkdir(parents=True)
    (src / "manifest.json").write_text("{}", encoding="utf-8")
    (src / "data.bin").write_bytes(b"hello")
    (src / "leaves" / "ndarray" / "blob.npy").write_bytes(b"arraybytes")
    payload_files = {p.resolve() for p in src.rglob("*") if p.is_file()}

    events: list[tuple[str, Path]] = []
    real_fsync_file = disk_mod._fsync_file
    real_rename = disk_mod.os.rename

    def _recording_fsync_file(path: Path) -> None:
        events.append(("fsync", path.resolve()))
        real_fsync_file(path)

    def _recording_rename(src_path: Path, dst_path: Path) -> None:
        events.append(("rename", src_path.resolve()))
        real_rename(src_path, dst_path)

    monkeypatch.setattr(disk_mod, "_fsync_file", _recording_fsync_file)
    monkeypatch.setattr(disk_mod.os, "rename", _recording_rename)

    store[key] = src

    fsynced = {path for kind, path in events if kind == "fsync"}
    assert payload_files <= fsynced  # every file in the tree was fsync'd
    rename_idx = next(i for i, (kind, _) in enumerate(events) if kind == "rename")
    last_fsync_idx = max(i for i, (kind, _) in enumerate(events) if kind == "fsync")
    assert last_fsync_idx < rename_idx  # all fsyncs happened before the publish rename
    assert (store[key] / "data.bin").read_bytes() == b"hello"


def test_disk_result_store_iterates_stored_results(tmp_path: Path) -> None:
    """``__iter__``/``__len__`` round-trip the base32-named payload dirs and skip noise."""
    store = DiskResultStore(tmp_path / "results")
    keys = {ResultHash(0xABCDEF), ResultHash(0x1234), ResultHash(0xFFFFFFFFFFFFFFFF)}
    for i, key in enumerate(sorted(keys)):
        src = tmp_path / f"payload_{i}"
        src.mkdir()
        (src / "data.bin").write_bytes(b"x")
        store[key] = src

    # Noise that must be ignored, not yielded or counted:
    sample = ResultHash(0x1234).b32()
    # (1) a leftover ``.trash`` dir from an interrupted delete -- dots and
    #     length put it outside the glob;
    (store.directory / sample[:2] / f"{sample}.deadbeef.trash").mkdir()
    # (2) a canonically-named dir under the wrong shard -- caught by the
    #     shard-prefix check (a broken check would inflate ``len``).
    (store.directory / "ZZ").mkdir()
    (store.directory / "ZZ" / sample).mkdir()

    assert set(store) == keys
    assert len(store) == len(keys)


def test_dependency_chain_leaves_no_dangling_mapping(tmp_path: Path) -> None:
    """A normal dependency-chain run commits payload+pointer for every node."""
    ws = DiskWorkspace(directory=str(tmp_path / "ws"))
    producer = Task(_produce, x=1)
    consumer = Task(_consume, v=producer.T)

    assert consumer.result(workspace=ws, compute_if_uncached=True, compute_uncached_deps=True) == 20

    for task in (producer, consumer):
        assert task.done(workspace=ws) is True
        result_hash = ws.get_result_hash(task)
        # Pointer and payload are both present -> nothing dangling.
        assert result_hash in ws.results.result_store
