# ruff: noqa: ANN001, PLR2004, S101, SLF001
"""Tests for the obstore-backed CloudWorkspace.

CloudWorkspace itself only accepts S3/GCS/Azure as production backends; for
hermetic unit tests we subclass it and inject obstore's ``MemoryStore``,
which fully implements the conditional-write primitives the lock and store
implementations rely on. That means these tests exercise the same code
paths the production cloud providers use.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

import obstore as obs
import pytest
from obstore.exceptions import GenericError, PreconditionError
from obstore.store import MemoryStore

from misen import SCRATCH_DIR, Task, meta
from misen.exceptions import LockUnavailableError, StorageError
from misen.utils.hashing import ResolvedTaskHash, ResultHash, TaskHash
from misen.utils.locks import LockLike, ObjectStoreLock
from misen.utils import locks as locks_module
from misen.utils.settings import Settings
from misen.workspace import Workspace
from misen.workspaces import cloud as cloud_mod
from misen.workspaces.cloud import CloudWorkspace, ObstoreMapping, ObstoreResultStore

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Test fixture: shared in-memory store keyed by bucket name so multiple
# CloudWorkspace instances pointing at the same logical "bucket" actually
# see each other's writes.
# ---------------------------------------------------------------------------

_shared_memory_stores: dict[str, MemoryStore] = {}


class _MemoryCloudWorkspace(CloudWorkspace):
    """CloudWorkspace variant that uses an in-memory obstore backend.

    ``backend`` and ``bucket`` are still required by the base struct but the
    bucket is interpreted as a name into a process-local registry of
    :class:`MemoryStore` instances. Two workspaces with the same ``bucket``
    share state.
    """

    def _build_store(self) -> Any:
        return _shared_memory_stores.setdefault(self.bucket, MemoryStore())


@meta(id="cloud_test_task_a", cache=True)
def cloud_test_task_a() -> int:
    """Test task that returns a constant integer."""
    return 11


@meta(id="cloud_test_task_b_for_filter", cache=True)
def cloud_test_task_b_for_filter() -> int:
    """Distinct test task used for work-unit filter tests."""
    return 22


@meta(id="cloud_test_writes_scratchdir", cache=True, exclude={"scratch_dir"})
def cloud_test_writes_scratchdir(scratch_dir: Path, value: str) -> int:
    """Write ``value`` to a fixed file inside the runtime scratch_dir."""
    (scratch_dir / "marker.txt").write_text(value)
    return len(value)


@meta(id="cloud_test_reads_scratchdir", cache=True, exclude={"scratch_dir"})
def cloud_test_reads_scratchdir(scratch_dir: Path) -> str:
    """Return the contents of a checkpoint file restored from durable storage."""
    f = scratch_dir / "preserved.txt"
    if f.exists():
        return f.read_text()
    return "fresh"


@meta(id="cloud_test_writes_scratchdir_cleanup", cache=True, exclude={"scratch_dir"})
def cloud_test_writes_scratchdir_cleanup(scratch_dir: Path) -> int:
    """Write to scratch_dir; the runtime cleans it up after the task succeeds."""
    (scratch_dir / "to_remove.txt").write_text("bye")
    return 7


@meta(id="cloud_test_writes_scratchdir_no_cache", cache=False, exclude={"scratch_dir"})
def cloud_test_writes_scratchdir_no_cache(scratch_dir: Path) -> int:
    """Non-cacheable task that writes to its ephemeral scratch_dir."""
    (scratch_dir / "ephemeral.txt").write_text("nope")
    return 1


# Module-level events for the during-execution sync test. Tests must reset
# them before use because the module is shared across tests.
_during_exec_can_finish = threading.Event()
_during_exec_observed = threading.Event()


@meta(id="cloud_test_pauses_in_scratchdir", cache=True, exclude={"scratch_dir"})
def cloud_test_pauses_in_scratchdir(scratch_dir: Path) -> int:
    """Write a checkpoint, pause for the test to inspect the bucket, then continue."""
    (scratch_dir / "early.txt").write_text("written-early")
    _during_exec_observed.set()
    if not _during_exec_can_finish.wait(timeout=10):
        msg = "test driver never released the during-execution gate"
        raise RuntimeError(msg)
    (scratch_dir / "late.txt").write_text("written-late")
    return 42


def _workspace(tmp_path, bucket: str) -> _MemoryCloudWorkspace:
    return _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        cache_dir=str(tmp_path / "cache"),
    )


def test_cloud_workspace_hash_caches_roundtrip(tmp_path) -> None:
    """Resolved/result hash caches roundtrip values through the store."""
    workspace = _workspace(tmp_path, "test-hash-caches")
    task = Task(cloud_test_task_a)

    assert workspace.get_resolved_hash(task) is None

    resolved = ResolvedTaskHash.from_object(("resolved", "a"))
    workspace.set_resolved_hash(task, resolved)
    assert workspace.get_resolved_hash(task) == resolved

    result_hash = ResultHash.from_object(("result", "a"))
    workspace.set_result_hash(task, result_hash)
    assert workspace.get_result_hash(task) == result_hash


def test_cloud_workspace_id_isolates_distinct_workspaces(tmp_path) -> None:
    """Distinct identities resolve to distinct cache subdirs under one base."""
    base = tmp_path / "cache"
    ws_default = _MemoryCloudWorkspace(backend="s3", bucket="b", cache_dir=str(base))
    ws_prefix = _MemoryCloudWorkspace(backend="s3", bucket="b", prefix="x", cache_dir=str(base))
    ws_endpoint = _MemoryCloudWorkspace(backend="s3", bucket="b", endpoint="https://r2.example", cache_dir=str(base))
    ws_region = _MemoryCloudWorkspace(backend="s3", bucket="b", s3_region="us-west-2", cache_dir=str(base))

    ids = {ws.workspace_id for ws in (ws_default, ws_prefix, ws_endpoint, ws_region)}
    assert len(ids) == 4
    for ws in (ws_default, ws_prefix, ws_endpoint, ws_region):
        assert ws._cache.parent == base
        assert ws._cache.name == ws.workspace_id

    # Same identity => same id => same cache subdir (safe sharing).
    twin = _MemoryCloudWorkspace(backend="s3", bucket="b", cache_dir=str(base))
    assert twin.workspace_id == ws_default.workspace_id
    assert twin._cache == ws_default._cache


def test_cloud_workspace_rejects_misapplied_locator_fields() -> None:
    """``s3_region`` and ``endpoint`` only apply to backends that support them."""
    with pytest.raises(ValueError, match="s3_region"):
        _MemoryCloudWorkspace(backend="gcs", bucket="b", s3_region="us-east-1")
    with pytest.raises(ValueError, match="s3_region"):
        _MemoryCloudWorkspace(backend="azure", bucket="b", s3_region="us-east-1")
    with pytest.raises(ValueError, match="endpoint"):
        _MemoryCloudWorkspace(backend="gcs", bucket="b", endpoint="https://example")


def test_cloud_workspace_rejects_duplicate_locator_in_config(tmp_path) -> None:
    """Setting the same locator key in both ``config`` and a dedicated field errors."""
    # _MemoryCloudWorkspace overrides _build_store, so use the real
    # CloudWorkspace path (validation runs in _build_store via __post_init__).
    with pytest.raises(ValueError, match="region"):
        CloudWorkspace(
            backend="s3",
            bucket="b",
            s3_region="us-east-1",
            config={"region": "us-east-1"},
            cache_dir=str(tmp_path / "cache"),
        )


def test_cloud_workspace_caches_persist_across_instances(tmp_path) -> None:
    """A second workspace with a fresh cache dir loads results from the bucket."""
    bucket = "test-persist"
    ws_a = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        cache_dir=str(tmp_path / "cache-a"),
    )
    task = Task(cloud_test_task_a)
    result = task.result(workspace=ws_a, compute_if_uncached=True, compute_uncached_deps=True)
    assert result == 11
    assert task.is_cached(workspace=ws_a)

    ws_b = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        cache_dir=str(tmp_path / "cache-b"),
    )
    assert task.is_cached(workspace=ws_b)
    assert task.result(workspace=ws_b) == 11


def test_cloud_workspace_log_uploaded_and_downloaded(tmp_path) -> None:
    """Log writes upload on close and a different cache dir downloads them."""
    bucket = "test-logs"
    ws_a = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        cache_dir=str(tmp_path / "cache-a"),
    )
    task = Task(cloud_test_task_a)
    resolved = ResolvedTaskHash.from_object(("log", "key"))
    ws_a.set_resolved_hash(task, resolved)

    log_path = ws_a.get_task_log(task=task, job_id="job-1")
    with log_path.open("a", encoding="utf-8") as f:
        f.write("hello-cloud\n")
    ws_a.finalize_task_log(task=task, job_id="job-1")

    with ws_a.read_task_log(task, job_id="job-1") as f:
        assert f.read().strip() == "hello-cloud"

    ws_b = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        cache_dir=str(tmp_path / "cache-b"),
    )
    ws_b.set_resolved_hash(task, resolved)
    with ws_b.read_task_log(task, job_id="job-1") as f:
        assert f.read().strip() == "hello-cloud"


def test_cloud_workspace_log_most_recent_when_no_job_id(tmp_path) -> None:
    """Read mode with no job_id picks the most recent log across local + remote."""
    workspace = _workspace(tmp_path, "test-log-recent")
    task = Task(cloud_test_task_a)
    workspace.set_resolved_hash(task, ResolvedTaskHash.from_object(("log", "recent")))

    first_path = workspace.get_task_log(task=task, job_id="first")
    with first_path.open("a", encoding="utf-8") as f:
        f.write("first\n")
    workspace.finalize_task_log(task=task, job_id="first")
    time.sleep(0.05)
    second_path = workspace.get_task_log(task=task, job_id="second")
    with second_path.open("a", encoding="utf-8") as f:
        f.write("second\n")
    workspace.finalize_task_log(task=task, job_id="second")

    with workspace.read_task_log(task) as f:
        assert f.read().strip() == "second"


def test_cloud_workspace_log_read_missing_raises(tmp_path) -> None:
    """Reading a log for a task that has never been logged raises FileNotFoundError."""
    workspace = _workspace(tmp_path, "test-log-missing")
    task = Task(cloud_test_task_a)
    workspace.set_resolved_hash(task, ResolvedTaskHash.from_object(("log", "missing")))
    with pytest.raises(FileNotFoundError):
        workspace.read_task_log(task)


def test_object_store_lock_implements_locklike_protocol(tmp_path) -> None:
    """ObjectStoreLock satisfies the LockLike protocol both structurally and nominally."""
    workspace = _workspace(tmp_path, "test-protocol")
    lock = workspace.lock(namespace="task", key="example")
    # Runtime structural check via @runtime_checkable Protocol.
    assert isinstance(lock, LockLike)
    # Nominal check: ObjectStoreLock explicitly subclasses LockLike.
    assert isinstance(lock, ObjectStoreLock)
    assert issubclass(ObjectStoreLock, LockLike)


def test_cloud_workspace_lock_serializes_holders(tmp_path) -> None:
    """A second holder cannot acquire while the first owns the lock."""
    workspace = _workspace(tmp_path, "test-lock-serialize")
    holder = workspace.lock(namespace="task", key="example")
    holder.acquire(blocking=True)
    try:
        contender = workspace.lock(namespace="task", key="example")
        with pytest.raises(LockUnavailableError):
            contender.acquire(blocking=False)
    finally:
        holder.release()


def test_object_store_lock_against_memory_store_conditional_writes() -> None:
    """ObjectStoreLock on MemoryStore exercises real put-if-absent + put-if-match."""
    store = MemoryStore()
    lock_a = ObjectStoreLock(store=store, key="locks/conditional", lifetime=60, refresh_interval=None)
    lock_a.acquire(blocking=True)
    assert lock_a.is_locked()

    lock_b = ObjectStoreLock(store=store, key="locks/conditional", lifetime=60, refresh_interval=None)
    with pytest.raises(LockUnavailableError):
        lock_b.acquire(blocking=False)

    lock_a.release()
    assert not lock_a.is_locked()

    lock_b.acquire(blocking=True)
    assert lock_b.is_locked()
    lock_b.release()


def test_object_store_lock_verifies_ownership_synchronously(monkeypatch: pytest.MonkeyPatch) -> None:
    store = MemoryStore()
    lock = ObjectStoreLock(store=store, key="locks/verify", lifetime=60, refresh_interval=None)
    lock.acquire(blocking=True)

    def lose_lease(*_args: object, **_kwargs: object) -> None:
        raise PreconditionError("stale token")

    monkeypatch.setattr(locks_module._obs, "put", lose_lease)

    assert not lock.is_locked()
    with pytest.raises(LockUnavailableError, match="Lost the lease"):
        lock.release()


def test_object_store_lock_surfaces_background_storage_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed refresh remains a storage failure during verification and release."""
    lock = ObjectStoreLock(store=MemoryStore(), key="locks/refresh", lifetime=60, refresh_interval=None)
    lock.acquire(blocking=True)

    def fail_refresh(*_args: object, **_kwargs: object) -> None:
        error = OSError("backend unavailable")
        raise error

    monkeypatch.setattr(locks_module._obs, "put", fail_refresh)
    lock._refresh_interval = 0.01
    lock._start_refresh()
    assert lock._thread is not None
    lock._thread.join(timeout=1)
    assert not lock._thread.is_alive()

    with pytest.raises(StorageError, match="Could not refresh"):
        lock.is_locked()
    with pytest.raises(StorageError, match="Could not refresh"):
        lock.release()


def test_object_store_lock_takes_over_after_lease_expiry() -> None:
    """A new holder takes over after the previous lease's expiry has elapsed."""
    store = MemoryStore()
    holder = ObjectStoreLock(store=store, key="locks/expiry", lifetime=0, refresh_interval=None)
    holder.acquire(blocking=True)
    time.sleep(0.05)

    contender = ObjectStoreLock(store=store, key="locks/expiry", lifetime=10, refresh_interval=None)
    contender.acquire(blocking=True, timeout=2)
    assert contender.is_locked()
    contender.release()


def test_object_store_lock_stale_release_does_not_delete_new_holder() -> None:
    """A stale holder's release cannot remove a lock that was taken over."""
    store = MemoryStore()
    stale = ObjectStoreLock(store=store, key="locks/stale-release", lifetime=0, refresh_interval=None)
    stale.acquire(blocking=True)
    time.sleep(0.01)

    holder = ObjectStoreLock(store=store, key="locks/stale-release", lifetime=60, refresh_interval=None)
    holder.acquire(blocking=True, timeout=2)

    stale.release()

    contender = ObjectStoreLock(store=store, key="locks/stale-release", lifetime=60, refresh_interval=None)
    with pytest.raises(LockUnavailableError):
        contender.acquire(blocking=False)

    holder.release()
    contender.acquire(blocking=True, timeout=2)
    contender.release()


def test_cloud_workspace_lock_blocks_other_thread(tmp_path) -> None:
    """A second thread blocks on lock acquire until the first releases."""
    workspace = _workspace(tmp_path, "test-lock-cross-thread")
    lock_a = workspace.lock(namespace="task", key="cross-thread")
    lock_b = workspace.lock(namespace="task", key="cross-thread")

    contender_acquired = threading.Event()

    def contender() -> None:
        lock_b.acquire(blocking=True, timeout=5)
        try:
            contender_acquired.set()
        finally:
            lock_b.release()

    lock_a.acquire(blocking=True)
    t = threading.Thread(target=contender, daemon=True)
    t.start()

    assert not contender_acquired.wait(timeout=0.2)
    lock_a.release()
    t.join(timeout=5.0)
    assert contender_acquired.is_set()


def test_obstore_mapping_iter_and_delete() -> None:
    """ObstoreMapping iteration returns all keys and __delitem__ raises on miss."""
    store = MemoryStore()
    mapping: ObstoreMapping[TaskHash, ResolvedTaskHash] = ObstoreMapping[TaskHash, ResolvedTaskHash](store, "resolved")
    keys = [TaskHash.from_object(("k", i)) for i in range(3)]
    values = [ResolvedTaskHash.from_object(("v", i)) for i in range(3)]
    for k, v in zip(keys, values, strict=True):
        mapping[k] = v
    assert set(iter(mapping)) == set(keys)
    assert len(mapping) == 3

    del mapping[keys[0]]
    assert keys[0] not in mapping
    with pytest.raises(KeyError):
        del mapping[keys[0]]


def test_obstore_mapping_corrupt_value_raises_storage_error() -> None:
    store = MemoryStore()
    mapping: ObstoreMapping[TaskHash, ResolvedTaskHash] = ObstoreMapping[TaskHash, ResolvedTaskHash](store, "resolved")
    key = TaskHash.from_object("corrupt")
    obs.put(store, f"resolved/{key.b32()}", b"corrupt")

    with pytest.raises(StorageError, match="corrupt or incompatible") as raised:
        _ = mapping[key]

    assert isinstance(raised.value.__cause__, ValueError)


def test_obstore_result_store_setitem_skips_when_present(tmp_path) -> None:
    """Re-setting a result hash is a no-op so existing payloads are preserved."""
    store = MemoryStore()
    cache_dir = tmp_path / "cache"
    rs = ObstoreResultStore(store, "results", cache_dir=cache_dir)
    rh = ResultHash.from_object(("res", 1))

    src = tmp_path / "src"
    src.mkdir()
    (src / "manifest.json").write_text("first")
    rs[rh] = src

    src2 = tmp_path / "src2"
    src2.mkdir()
    (src2 / "manifest.json").write_text("second")
    rs[rh] = src2

    materialized = rs[rh]
    assert (materialized / "manifest.json").read_text() == "first"


def test_obstore_result_store_ignores_uncommitted_payloads(tmp_path) -> None:
    """A partial remote upload is invisible until manifest.json is present."""
    store = MemoryStore()
    cache_dir = tmp_path / "cache"
    rs = ObstoreResultStore(store, "results", cache_dir=cache_dir)
    rh = ResultHash.from_object(("res", "partial"))

    obs.put(store, f"results/{rh.b32()}/leaves/data.bin", b"partial", mode="overwrite")

    assert rh not in rs
    assert list(rs) == []
    with pytest.raises(KeyError):
        _ = rs[rh]

    with pytest.raises(KeyError):
        del rs[rh]


def test_obstore_result_store_isolates_interleaved_generations(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale upload cannot overwrite files from the committed generation."""
    store = MemoryStore()
    rs = ObstoreResultStore(store, "results", cache_dir=tmp_path / "cache")
    rh = ResultHash.from_object(("res", "interleaved"))

    stale = tmp_path / "stale"
    winner = tmp_path / "winner"
    for directory, label in ((stale, "stale"), (winner, "winner")):
        directory.mkdir()
        (directory / "a.bin").write_text(f"{label}-a")
        (directory / "manifest.json").write_text(f"{label}-manifest")
        (directory / "z.bin").write_text(f"{label}-z")

    real_put = cloud_mod.obs.put
    interleaved = False

    def put_with_winner(store_arg: Any, path: str, payload: Any, **kwargs: Any) -> Any:
        nonlocal interleaved
        if not interleaved and path.endswith("/z.bin"):
            interleaved = True
            rs.commit(rh, winner, before_commit=lambda: None)
        return real_put(store_arg, path, payload, **kwargs)

    monkeypatch.setattr(cloud_mod.obs, "put", put_with_winner)

    def lost_ownership() -> None:
        msg = "lost runtime lease"
        raise LockUnavailableError(msg)

    with pytest.raises(LockUnavailableError, match="lost runtime lease"):
        rs.commit(rh, stale, before_commit=lost_ownership)

    materialized = rs[rh]
    assert {path.name: path.read_text() for path in materialized.iterdir()} == {
        "a.bin": "winner-a",
        "manifest.json": "winner-manifest",
        "z.bin": "winner-z",
    }
    assert list(rs) == [rh]


def test_obstore_result_store_reads_and_deletes_legacy_layout(tmp_path) -> None:
    """Existing root-level payloads remain readable and fully deletable."""
    store = MemoryStore()
    rs = ObstoreResultStore(store, "results", cache_dir=tmp_path / "cache")
    rh = ResultHash.from_object(("res", "legacy"))
    prefix = f"results/{rh.b32()}"
    obs.put(store, f"{prefix}/manifest.json", b"legacy-manifest")
    obs.put(store, f"{prefix}/leaves/data.bin", b"legacy-data")
    obs.put(store, f"{prefix}/.builds/ORPHAN/manifest.json", b"orphan")

    materialized = rs[rh]
    assert (materialized / "manifest.json").read_bytes() == b"legacy-manifest"
    assert (materialized / "leaves/data.bin").read_bytes() == b"legacy-data"
    assert not (materialized / ".builds").exists()
    assert list(rs) == [rh]

    del rs[rh]
    assert list(obs.list(store, prefix=f"{prefix}/")) == []


def test_obstore_mapping_fenced_commit_does_not_overwrite_winner() -> None:
    """A stale create cannot overwrite a pointer published during its fence."""
    store = MemoryStore()
    mapping = ObstoreMapping[ResolvedTaskHash, ResultHash](store, "result_hashes")
    key = ResolvedTaskHash(1)
    stale, winner = ResultHash(1), ResultHash(2)

    with pytest.raises(LockUnavailableError, match="Another writer"):
        mapping.commit(key, stale, before_commit=lambda: mapping.__setitem__(key, winner))

    assert mapping[key] == winner


def test_workspace_auto_resolves_cloud_from_toml(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``Workspace.auto`` resolves the ``cloud`` alias with the new fields."""
    config = tmp_path / "misen.toml"
    config.write_text(
        f"""[workspace]
type = "cloud"
backend = "s3"
bucket = "auto-resolved"
cache_dir = "{tmp_path / "cache"}"
"""
    )
    monkeypatch.setitem(
        Workspace._config_aliases,
        "cloud",
        f"{_MemoryCloudWorkspace.__module__}:{_MemoryCloudWorkspace.__qualname__}",
    )
    workspace = Workspace.auto(settings=Settings(config_file=config))
    assert isinstance(workspace, CloudWorkspace)
    assert workspace.backend == "s3"
    assert workspace.bucket == "auto-resolved"


def _work_unit_for(task_callable: Any) -> Any:
    from misen.utils.work_unit import WorkUnit

    return WorkUnit(root=Task(task_callable), dependencies=set())


def _store_paths(workspace: _MemoryCloudWorkspace, prefix: str = "") -> set[str]:
    return {entry["path"] for batch in obs.list(workspace._store, prefix=prefix) for entry in batch}


def test_cloud_workspace_streaming_job_log_uploads_on_exit(tmp_path) -> None:
    """Exiting streaming_job_log uploads the file's final state to the bucket."""
    workspace = _workspace(tmp_path, "test-stream-exit")
    work_unit = _work_unit_for(cloud_test_task_a)
    log_path = workspace.get_job_log(job_id="job-1", work_unit=work_unit)

    with workspace.streaming_job_log(log_path):
        log_path.write_text("job output")

    other = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-stream-exit",
        cache_dir=str(tmp_path / "cache-2"),
    )
    paths = list(other.job_log_iter(work_unit=work_unit))
    assert len(paths) == 1
    assert paths[0].read_text() == "job output"


def test_cloud_workspace_job_log_chunks_are_compacted_on_finalize(tmp_path) -> None:
    """Live job-log chunks are removed once the canonical log is uploaded."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-job-chunk-compaction",
        cache_dir=str(tmp_path / "cache"),
        log_flush_interval_s=0.05,
    )
    work_unit = _work_unit_for(cloud_test_task_a)
    log_path = workspace.get_job_log(job_id="job-1", work_unit=work_unit)
    remote_key = workspace._job_log_remote_key(log_path)

    with workspace.streaming_job_log(log_path):
        log_path.write_text("chunked output\n")
        time.sleep(0.25)
        paths = _store_paths(workspace, "job_logs")
        assert any(path.startswith(f"{remote_key}.chunks/") for path in paths)
        assert f"{remote_key}.state.json" in paths
        assert remote_key not in paths

    paths = _store_paths(workspace, "job_logs")
    assert remote_key in paths
    assert not any(path.startswith(f"{remote_key}.chunks/") for path in paths)
    assert f"{remote_key}.state.json" not in paths


def test_cloud_workspace_prefers_compacted_log_over_stale_chunks(tmp_path) -> None:
    """A compacted log wins if stale live chunks were not deleted."""
    bucket = "test-prefers-final"
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        cache_dir=str(tmp_path / "cache-a"),
    )
    work_unit = _work_unit_for(cloud_test_task_a)
    log_path = workspace.get_job_log(job_id="job-1", work_unit=work_unit)
    remote_key = workspace._job_log_remote_key(log_path)

    other = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        cache_dir=str(tmp_path / "cache-b"),
    )

    obs.put(
        workspace._store,
        f"{remote_key}.chunks/00000000000000000000.chunk",
        b"stale bytes",
        mode="overwrite",
    )
    obs.put(
        workspace._store,
        f"{remote_key}.state.json",
        b'{"offset": 11, "closed": false}',
        mode="overwrite",
    )
    paths = list(other.job_log_iter(work_unit=work_unit))
    assert len(paths) == 1
    assert paths[0].read_text() == "stale bytes"

    time.sleep(0.05)
    obs.put(workspace._store, remote_key, b"final bytes", mode="overwrite")

    paths = list(other.job_log_iter(work_unit=work_unit))
    assert len(paths) == 1
    assert paths[0].read_text() == "final bytes"


def test_cloud_workspace_ignores_incomplete_live_chunks(tmp_path) -> None:
    """A state marker cannot materialize fewer live bytes than it advertises."""
    bucket = "test-incomplete-chunks"
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        cache_dir=str(tmp_path / "cache-a"),
    )
    work_unit = _work_unit_for(cloud_test_task_a)
    log_path = workspace.get_job_log(job_id="job-1", work_unit=work_unit)
    remote_key = workspace._job_log_remote_key(log_path)

    obs.put(workspace._store, remote_key, b"last complete\n", mode="overwrite")
    time.sleep(0.05)
    obs.put(
        workspace._store,
        f"{remote_key}.chunks/00000000000000000000.chunk",
        b"partial\n",
        mode="overwrite",
    )
    obs.put(
        workspace._store,
        f"{remote_key}.state.json",
        b'{"offset": 99, "closed": false}',
        mode="overwrite",
    )

    other = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        cache_dir=str(tmp_path / "cache-b"),
    )

    paths = list(other.job_log_iter(work_unit=work_unit))
    assert len(paths) == 1
    assert paths[0].read_text() == "last complete\n"


def test_cloud_workspace_refresh_does_not_overwrite_active_local_log(tmp_path) -> None:
    """Remote refresh avoids local logs that this workspace is uploading."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-active-log-refresh",
        cache_dir=str(tmp_path / "cache"),
        log_flush_interval_s=60,
    )
    work_unit = _work_unit_for(cloud_test_task_a)
    log_path = workspace.get_job_log(job_id="job-1", work_unit=work_unit)
    remote_key = workspace._job_log_remote_key(log_path)

    log_path.write_text("local in progress\n")
    with workspace.streaming_job_log(log_path):
        obs.put(workspace._store, remote_key, b"remote stale\n", mode="overwrite")
        workspace._refresh_log_local(remote_key, log_path)

    assert log_path.read_text() == "local in progress\n"


def test_cloud_workspace_streaming_job_log_missing_file_is_noop(tmp_path) -> None:
    """A missing local file does not raise (job died before producing output)."""
    workspace = _workspace(tmp_path, "test-stream-missing")
    work_unit = _work_unit_for(cloud_test_task_a)
    log_path = workspace.get_job_log(job_id="never-ran", work_unit=work_unit)
    with workspace.streaming_job_log(log_path):
        # Worker exited without writing anything.
        pass


def test_cloud_workspace_get_job_log_is_pure(tmp_path) -> None:
    """get_job_log only returns a path; it does not start streaming."""
    workspace = _workspace(tmp_path, "test-get-pure")
    work_unit = _work_unit_for(cloud_test_task_a)
    workspace.get_job_log(job_id="job-pure", work_unit=work_unit)
    # No streaming context entered, so no live uploader registered.
    assert not workspace._live_log_uploaders


def test_cloud_workspace_finalize_job_log_captures_post_streaming_writes(tmp_path) -> None:
    """finalize_job_log uploads anything written after the streaming context closed.

    Mirrors the SLURM lifecycle: the worker's streaming context handles
    writes during execution; the controller appends an epilogue after the
    wrapped command exits; the parent's finalize captures it.
    """
    bucket = "test-finalize-shot"
    ws_a = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        cache_dir=str(tmp_path / "cache-a"),
    )
    work_unit = _work_unit_for(cloud_test_task_a)
    log_path = ws_a.get_job_log(job_id="job-1", work_unit=work_unit)

    log_path.write_text("worker output\n")
    with ws_a.streaming_job_log(log_path):
        pass  # Streaming exits, final upload captures "worker output\n"

    # Simulate SLURM appending an epilogue after the wrapped command exited.
    log_path.write_text("worker output\nslurm epilogue\n")

    # Without finalize, a second client only sees the streaming-time content.
    ws_b = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        cache_dir=str(tmp_path / "cache-b"),
    )
    paths = list(ws_b.job_log_iter(work_unit=work_unit))
    assert paths[0].read_text() == "worker output\n"

    # After finalize, the bucket reflects the file's current state.
    ws_a.finalize_job_log(log_path)

    ws_c = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        cache_dir=str(tmp_path / "cache-c"),
    )
    paths = list(ws_c.job_log_iter(work_unit=work_unit))
    assert paths[0].read_text() == "worker output\nslurm epilogue\n"


def test_cloud_workspace_finalize_job_log_missing_file_is_noop(tmp_path) -> None:
    """finalize_job_log does not raise when the local file is absent."""
    workspace = _workspace(tmp_path, "test-finalize-missing")
    work_unit = _work_unit_for(cloud_test_task_a)
    log_path = workspace.get_job_log(job_id="never-ran", work_unit=work_unit)
    workspace.finalize_job_log(log_path)


def test_cloud_workspace_finalize_job_log_is_idempotent(tmp_path) -> None:
    """Repeated finalize calls are safe and reflect the current file state."""
    workspace = _workspace(tmp_path, "test-finalize-idem")
    work_unit = _work_unit_for(cloud_test_task_a)
    log_path = workspace.get_job_log(job_id="job-1", work_unit=work_unit)
    log_path.write_text("content")
    workspace.finalize_job_log(log_path)
    workspace.finalize_job_log(log_path)

    other = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-finalize-idem",
        cache_dir=str(tmp_path / "cache-2"),
    )
    paths = list(other.job_log_iter(work_unit=work_unit))
    assert paths[0].read_text() == "content"


def test_cloud_workspace_final_log_failure_raises_storage_error(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed canonical log upload is observable and retains its cause."""
    workspace = _workspace(tmp_path, "test-final-log-failure")
    work_unit = _work_unit_for(cloud_test_task_a)
    log_path = workspace.get_job_log(job_id="job-1", work_unit=work_unit)
    log_path.write_text("content")
    failure = GenericError("object store unavailable")

    def fail_put(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        raise failure

    monkeypatch.setattr(obs, "put", fail_put)

    with pytest.raises(StorageError, match="Could not finalize log") as exc_info:
        workspace.finalize_job_log(log_path)

    assert exc_info.value.__cause__ is failure


def test_cloud_workspace_final_log_does_not_wrap_programmer_error(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected implementation errors remain visible during finalization."""
    workspace = _workspace(tmp_path, "test-final-log-programmer-error")
    work_unit = _work_unit_for(cloud_test_task_a)
    log_path = workspace.get_job_log(job_id="job-1", work_unit=work_unit)
    log_path.write_text("content")

    def fail_put(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        msg = "uploader bug"
        raise AssertionError(msg)

    monkeypatch.setattr(obs, "put", fail_put)

    with pytest.raises(AssertionError, match="uploader bug"):
        workspace.finalize_job_log(log_path)


def test_cloud_workspace_job_log_iter_filters_by_work_unit(tmp_path) -> None:
    """job_log_iter merges local + remote and respects the work_unit filter."""
    workspace = _workspace(tmp_path, "test-jli-filter")
    wu_a = _work_unit_for(cloud_test_task_a)
    wu_b = _work_unit_for(cloud_test_task_b_for_filter)

    for job_id, wu, body in (("a1", wu_a, "a1"), ("a2", wu_a, "a2"), ("b1", wu_b, "b1")):
        log_path = workspace.get_job_log(job_id=job_id, work_unit=wu)
        with workspace.streaming_job_log(log_path):
            log_path.write_text(body)

    # Fresh cache dir ensures the merge fetches everything from the bucket.
    other = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-jli-filter",
        cache_dir=str(tmp_path / "cache-2"),
    )
    a_logs = {p.read_text() for p in other.job_log_iter(work_unit=wu_a)}
    all_logs = {p.read_text() for p in other.job_log_iter()}
    assert a_logs == {"a1", "a2"}
    assert all_logs == {"a1", "a2", "b1"}


def test_cloud_workspace_task_log_live_streamed_to_bucket(tmp_path) -> None:
    """A second client sees task-log writes before the writer is closed."""
    bucket = "test-task-live"
    ws_a = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        cache_dir=str(tmp_path / "cache-a"),
        log_flush_interval_s=0.05,
    )
    task = Task(cloud_test_task_a)
    ws_a.set_resolved_hash(task, ResolvedTaskHash.from_object(("live", "task")))

    log_path = ws_a.get_task_log(task=task, job_id="live-job")
    fp = log_path.open("a", buffering=1, encoding="utf-8")
    try:
        fp.write("line-1\n")
        fp.flush()
        # Wait for at least one live-upload tick.
        time.sleep(0.25)

        ws_b = _MemoryCloudWorkspace(
            backend="s3",
            bucket=bucket,
            cache_dir=str(tmp_path / "cache-b"),
            log_flush_interval_s=0.05,
        )
        ws_b.set_resolved_hash(task, ResolvedTaskHash.from_object(("live", "task")))
        with ws_b.read_task_log(task, job_id="live-job") as f:
            mid_state = f.read()
        assert mid_state.strip() == "line-1"

        fp.write("line-2\n")
        fp.flush()
        time.sleep(0.25)

        with ws_b.read_task_log(task, job_id="live-job") as f:
            refreshed_state = f.read()
        assert refreshed_state.splitlines() == ["line-1", "line-2"]
    finally:
        fp.close()
        ws_a.finalize_task_log(task=task, job_id="live-job")

    ws_c = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        cache_dir=str(tmp_path / "cache-c"),
    )
    ws_c.set_resolved_hash(task, ResolvedTaskHash.from_object(("live", "task")))
    with ws_c.read_task_log(task, job_id="live-job") as f:
        final_state = f.read()
    assert final_state.splitlines() == ["line-1", "line-2"]


def test_cloud_workspace_job_log_live_streamed_to_bucket(tmp_path) -> None:
    """A second client sees job-log writes while streaming_job_log is open."""
    bucket = "test-job-live"
    ws_a = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        cache_dir=str(tmp_path / "cache-a"),
        log_flush_interval_s=0.05,
    )
    work_unit = _work_unit_for(cloud_test_task_a)
    log_path = ws_a.get_job_log(job_id="job-live", work_unit=work_unit)

    with ws_a.streaming_job_log(log_path):
        log_path.write_text("partial output\n")
        time.sleep(0.25)  # Wait for at least one live-upload tick.

        ws_b = _MemoryCloudWorkspace(
            backend="s3",
            bucket=bucket,
            cache_dir=str(tmp_path / "cache-b"),
        )
        paths = list(ws_b.job_log_iter(work_unit=work_unit))
        assert len(paths) == 1
        assert paths[0].read_text().strip() == "partial output"

        log_path.write_text("partial output\nmore output\n")
        time.sleep(0.25)

        paths = list(ws_b.job_log_iter(work_unit=work_unit))
        assert len(paths) == 1
        assert paths[0].read_text().splitlines() == ["partial output", "more output"]

    ws_c = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        cache_dir=str(tmp_path / "cache-c"),
    )
    paths = list(ws_c.job_log_iter(work_unit=work_unit))
    assert len(paths) == 1
    assert paths[0].read_text().splitlines() == ["partial output", "more output"]


def test_live_log_background_upload_retries_operational_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient background upload error is logged and retried on the next tick."""
    workspace = _workspace(tmp_path, "test-live-log-retry")
    local_path = tmp_path / "live.log"
    local_path.write_text("content")
    remote_key = workspace._under("job_logs", "live.log")
    real_put = obs.put
    attempts = 0

    def flaky_put(store: Any, path: str, file: Any, **kwargs: Any) -> Any:
        nonlocal attempts
        if path.startswith(f"{remote_key}.chunks/"):
            attempts += 1
            if attempts == 1:
                msg = "temporary upload failure"
                raise OSError(msg)
        return real_put(store, path, file, **kwargs)

    monkeypatch.setattr(obs, "put", flaky_put)
    uploader = cloud_mod._LiveLogUploader(workspace._store, local_path, remote_key, 0.02)
    uploader.start()
    try:
        time.sleep(0.15)
    finally:
        uploader.stop(final_upload=False)

    assert attempts >= 2
    assert f"{remote_key}.state.json" in _store_paths(workspace, "job_logs")


def test_cloud_workspace_read_does_not_truncate_sibling_writer_task_log(tmp_path) -> None:
    """A read in one workspace must not clobber a sibling writer on the same FS.

    With LocalExecutor (and SLURM on shared NFS), the orchestrator and the
    worker subprocess share ``.cache/misen`` via cwd. A cache refresh that
    materialized downloaded chunks into a path the worker was still
    appending to would orphan the worker's open inode and lose every byte
    written between the last chunk upload and the rename. Regression test
    for that bug.
    """
    bucket = "test-sibling-task-log"
    cache = tmp_path / "cache-shared"
    writer_ws = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        cache_dir=str(cache),
        log_flush_interval_s=60,  # Long; chunk + state are published manually below.
    )
    task = Task(cloud_test_task_a)
    writer_ws.set_resolved_hash(task, ResolvedTaskHash.from_object(("sibling", "task")))

    log_path = writer_ws.get_task_log(task=task, job_id="job-sib")
    remote_key = writer_ws._under("task_logs", task.task_hash().b32(), "job-sib.log")

    fp = log_path.open("a", buffering=1, encoding="utf-8")
    try:
        fp.write("first-chunk\n")
        fp.flush()

        # Publish a chunk + state, simulating what the writer's live uploader
        # would have pushed. We do this by hand to keep the test
        # deterministic instead of racing with a background thread.
        obs.put(
            writer_ws._store,
            f"{remote_key}.chunks/00000000000000000000.chunk",
            b"first-chunk\n",
            mode="overwrite",
        )
        obs.put(
            writer_ws._store,
            f"{remote_key}.state.json",
            b'{"offset": 12, "closed": false}',
            mode="overwrite",
        )

        # Writer continues. Now ``local_size > state_info.offset`` -- this
        # is the gap where a refresh would (pre-fix) rename the writer's
        # file to a truncated copy.
        fp.write("second-chunk\n")
        fp.flush()

        # Reader in a separate workspace instance with the same cache:
        # simulates orchestrator-vs-subprocess (LocalExecutor). The
        # orchestrator's ``_live_log_uploaders`` dict does NOT contain the
        # writer path (a different process owns it), so the in-process
        # guard inside ``_refresh_log_local`` cannot save us here.
        reader_ws = _MemoryCloudWorkspace(
            backend="s3",
            bucket=bucket,
            cache_dir=str(cache),
        )
        reader_ws.set_resolved_hash(task, ResolvedTaskHash.from_object(("sibling", "task")))
        with reader_ws.read_task_log(task, job_id="job-sib") as f:
            f.read()

        # Pre-fix, the read above renamed the writer's file. This write
        # lands in the orphaned inode and is invisible to compact.
        fp.write("third-chunk\n")
        fp.flush()
    finally:
        fp.close()
        writer_ws.finalize_task_log(task=task, job_id="job-sib")

    # A fresh client (no shared cache) reads via the compacted .log blob.
    # If the read had clobbered the writer file, compact would have
    # uploaded only the truncated 12-byte chunk content.
    fresh = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        cache_dir=str(tmp_path / "cache-fresh"),
    )
    fresh.set_resolved_hash(task, ResolvedTaskHash.from_object(("sibling", "task")))
    with fresh.read_task_log(task, job_id="job-sib") as f:
        final = f.read()
    assert final.splitlines() == ["first-chunk", "second-chunk", "third-chunk"]


def test_cloud_workspace_close_stops_live_uploaders(tmp_path) -> None:
    """Closing the workspace stops any outstanding live-upload threads."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-close-stops",
        cache_dir=str(tmp_path / "cache"),
        log_flush_interval_s=0.05,
    )
    work_unit = _work_unit_for(cloud_test_task_a)
    log_path = workspace.get_job_log(job_id="job-x", work_unit=work_unit)
    # Manually start streaming without entering the context (simulating
    # a worker that died mid-execution and never reached __exit__).
    workspace._start_live_upload(log_path, workspace._job_log_remote_key(log_path))

    assert workspace._live_log_uploaders
    workspace.close()
    assert not workspace._live_log_uploaders


def test_cloud_workspace_producer_finalizer_uploads_writes_after_close(tmp_path) -> None:
    """A producer still owns the final commit when workspace close races it."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-close-producer-finalizer",
        cache_dir=str(tmp_path / "cache"),
        log_flush_interval_s=60,
    )
    work_unit = _work_unit_for(cloud_test_task_a)
    log_path = workspace.get_job_log(job_id="job-race", work_unit=work_unit)

    with workspace.streaming_job_log(log_path):
        log_path.write_text("before-close\n")
        workspace.close()
        with log_path.open("a") as fp:
            fp.write("after-close\n")

    remote_key = workspace._job_log_remote_key(log_path)
    assert bytes(obs.get(workspace._store, remote_key).bytes()) == b"before-close\nafter-close\n"


def test_cloud_workspace_producer_finalizer_uploads_after_failed_close(tmp_path) -> None:
    """A different close failure cannot suppress a producer's final upload."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-failed-close-producer-finalizer",
        cache_dir=str(tmp_path / "cache"),
        log_flush_interval_s=60,
    )
    work_unit = _work_unit_for(cloud_test_task_a)
    log_path = workspace.get_job_log(job_id="job-race", work_unit=work_unit)

    class FailingSync:
        def stop(self, *, final_upload: bool) -> None:
            assert final_upload
            raise StorageError("scratch finalization failed")

    workspace._scratch_dir_syncs["failing"] = FailingSync()  # type: ignore[assignment]
    with workspace.streaming_job_log(log_path):
        log_path.write_text("before-close\n")
        with pytest.raises(StorageError, match="scratch finalization failed"):
            workspace.close()
        with log_path.open("a") as fp:
            fp.write("after-close\n")

    remote_key = workspace._job_log_remote_key(log_path)
    assert bytes(obs.get(workspace._store, remote_key).bytes()) == b"before-close\nafter-close\n"

    workspace._scratch_dir_syncs.clear()
    workspace.close()


def test_cloud_workspace_results_iter(tmp_path) -> None:
    """The result store iterates over all stored result hashes."""
    workspace = _workspace(tmp_path, "test-results-iter")
    task = Task(cloud_test_task_a)
    task.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True)
    iter_count = sum(1 for _ in iter(workspace.results.result_store))
    assert iter_count == 1


def _scratch_dir_remote_paths(workspace: _MemoryCloudWorkspace, task: Task) -> set[str]:
    prefix = workspace._scratch_dir_remote_prefix(task) + "/"
    return {entry["path"] for batch in obs.list(workspace._store, prefix=prefix) for entry in batch}


def test_cloud_workspace_scratch_dir_restored_from_bucket(tmp_path) -> None:
    """A pre-existing cloud snapshot is restored into the local scratch_dir before the task runs."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-scratchdir-restored",
        cache_dir=str(tmp_path / "cache"),
    )
    task = Task(cloud_test_reads_scratchdir, SCRATCH_DIR)

    remote_prefix = workspace._scratch_dir_remote_prefix(task)
    obs.put(workspace._store, f"{remote_prefix}/preserved.txt", b"from-bucket", mode="overwrite")

    result = task.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True)
    assert result == "from-bucket"


def test_cloud_workspace_scratch_dir_synced_during_execution(tmp_path) -> None:
    """Files written during a long-running task land in the bucket before completion."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-scratchdir-during",
        cache_dir=str(tmp_path / "cache"),
        scratch_dir_sync_interval_s=0.05,
    )
    task = Task(cloud_test_pauses_in_scratchdir, SCRATCH_DIR)
    _during_exec_can_finish.clear()
    _during_exec_observed.clear()

    runner_error: list[BaseException] = []

    def run() -> None:
        try:
            task.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True)
        except BaseException as e:  # noqa: BLE001
            runner_error.append(e)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    try:
        assert _during_exec_observed.wait(timeout=5)
        # Wait long enough for at least one background sync tick to upload
        # ``early.txt`` while the task is still paused.
        time.sleep(0.3)
        paths = _scratch_dir_remote_paths(workspace, task)
        assert any(p.endswith("/early.txt") for p in paths)
        assert not any(p.endswith("/late.txt") for p in paths)
    finally:
        _during_exec_can_finish.set()
        runner.join(timeout=10)

    assert not runner_error, runner_error[0]
    # After successful completion the runtime cleans up scratch_dir
    # (local + bucket), so all synced files are gone.
    assert _scratch_dir_remote_paths(workspace, task) == set()


def test_cloud_workspace_scratch_dir_cleanup_removes_local_and_remote(tmp_path) -> None:
    """A successful cacheable task's scratch_dir is removed from both local cache and bucket."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-scratchdir-cleanup",
        cache_dir=str(tmp_path / "cache"),
    )
    task = Task(cloud_test_writes_scratchdir_cleanup, SCRATCH_DIR)
    assert task.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True) == 7

    paths = _scratch_dir_remote_paths(workspace, task)
    assert paths == set()
    local_path = workspace._cache / "scratch" / task.resolved_hash(workspace=workspace).b32()
    assert not local_path.exists()


def test_cloud_workspace_remove_scratch_dir_when_idle(tmp_path) -> None:
    """``remove_scratch_dir`` works when no sync session is currently active."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-scratchdir-remove-idle",
        cache_dir=str(tmp_path / "cache"),
    )
    task = Task(cloud_test_writes_scratchdir, SCRATCH_DIR, "data")
    task.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True)

    workspace.remove_scratch_dir(task=task)
    assert _scratch_dir_remote_paths(workspace, task) == set()
    local_path = workspace._cache / "scratch" / task.resolved_hash(workspace=workspace).b32()
    assert not local_path.exists()


def test_cloud_workspace_remove_scratch_dir_rejects_non_cacheable(tmp_path) -> None:
    """``remove_scratch_dir`` refuses non-cacheable tasks."""
    workspace = _workspace(tmp_path, "test-scratchdir-remove-rejects")
    task = Task(cloud_test_writes_scratchdir_no_cache, SCRATCH_DIR)
    with pytest.raises(RuntimeError, match="cannot use workspace scratch_dir"):
        workspace.remove_scratch_dir(task=task)


def test_cloud_workspace_scratch_dir_not_synced_for_non_cacheable(tmp_path) -> None:
    """Non-cacheable tasks use ephemeral local scratch_dirs and never publish to the bucket."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-scratchdir-non-cache",
        cache_dir=str(tmp_path / "cache"),
    )
    task = Task(cloud_test_writes_scratchdir_no_cache, SCRATCH_DIR)
    task.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True)

    scratch_dir_paths = {
        entry["path"]
        for batch in obs.list(workspace._store, prefix=workspace._under("scratch_dirs") + "/")
        for entry in batch
    }
    assert scratch_dir_paths == set()


def test_cloud_workspace_close_stops_scratch_dir_syncs(tmp_path) -> None:
    """Closing the workspace tears down any outstanding scratch_dir sync threads."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-scratchdir-close",
        cache_dir=str(tmp_path / "cache"),
        scratch_dir_sync_interval_s=0.05,
    )
    task = Task(cloud_test_writes_scratchdir, SCRATCH_DIR, "data")
    workspace.start_scratch_dir_sync(task=task)
    assert workspace._scratch_dir_syncs

    workspace.close()
    assert not workspace._scratch_dir_syncs


def test_cloud_workspace_scratch_finalizer_mirrors_changes_after_close(tmp_path) -> None:
    """Task finalization publishes scratch changes made after workspace close."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-scratchdir-close-race",
        cache_dir=str(tmp_path / "cache"),
        scratch_dir_sync_interval_s=60,
    )
    task = Task(cloud_test_writes_scratchdir, SCRATCH_DIR, "data")
    workspace.start_scratch_dir_sync(task=task)
    local_dir = workspace._get_scratch_dir(task)
    obsolete = local_dir / "obsolete.txt"
    obsolete.write_text("old")

    workspace.close()
    obsolete.unlink()
    (local_dir / "final.txt").write_text("new")
    workspace.finalize_scratch_dir(task=task)

    remote_prefix = workspace._scratch_dir_remote_prefix(task)
    assert _scratch_dir_remote_paths(workspace, task) == {f"{remote_prefix}/final.txt"}
    assert bytes(obs.get(workspace._store, f"{remote_prefix}/final.txt").bytes()) == b"new"


def test_cloud_workspace_scratch_dir_sync_drops_deleted_files(tmp_path) -> None:
    """Files removed locally during execution are removed from the bucket on the next tick."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-scratchdir-deletion",
        cache_dir=str(tmp_path / "cache"),
        scratch_dir_sync_interval_s=0.05,
    )
    task = Task(cloud_test_writes_scratchdir, SCRATCH_DIR, "value")
    workspace.start_scratch_dir_sync(task=task)
    try:
        local_dir = workspace._get_scratch_dir(task)
        (local_dir / "transient.txt").write_text("temporary")
        time.sleep(0.25)
        paths = _scratch_dir_remote_paths(workspace, task)
        assert any(p.endswith("/transient.txt") for p in paths)

        (local_dir / "transient.txt").unlink()
        time.sleep(0.25)
        paths = _scratch_dir_remote_paths(workspace, task)
        assert not any(p.endswith("/transient.txt") for p in paths)
    finally:
        workspace.finalize_scratch_dir(task=task)


def test_cloud_workspace_final_scratch_sync_failure_raises_storage_error(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed terminal scratch upload propagates with the original cause."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-scratch-final-failure",
        cache_dir=str(tmp_path / "cache"),
        scratch_dir_sync_interval_s=60,
    )
    task = Task(cloud_test_writes_scratchdir, SCRATCH_DIR, "value")
    workspace.start_scratch_dir_sync(task=task)
    local_dir = workspace._get_scratch_dir(task)
    (local_dir / "checkpoint.txt").write_text("checkpoint")
    failure = OSError("disk read failed")

    def fail_put(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        raise failure

    monkeypatch.setattr(obs, "put", fail_put)

    with pytest.raises(StorageError, match="Could not finalize scratch directory") as exc_info:
        workspace.finalize_scratch_dir(task=task)

    assert exc_info.value.__cause__ is failure


def test_scratch_background_sync_retries_operational_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Periodic scratch syncing remains best-effort across transient failures."""
    workspace = _workspace(tmp_path, "test-scratch-background-retry")
    local_dir = tmp_path / "scratch"
    local_dir.mkdir()
    (local_dir / "checkpoint.txt").write_text("checkpoint")
    remote_prefix = workspace._under("scratch_dirs", "retry")
    real_put = obs.put
    attempts = 0

    def flaky_put(store: Any, path: str, file: Any, **kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            msg = "temporary scratch upload failure"
            raise OSError(msg)
        return real_put(store, path, file, **kwargs)

    monkeypatch.setattr(obs, "put", flaky_put)
    sync = cloud_mod._ScratchDirSync(workspace._store, local_dir, remote_prefix, 0.02)
    sync.start()
    try:
        time.sleep(0.15)
    finally:
        sync.stop(final_upload=False)

    assert attempts >= 2
    assert f"{remote_prefix}/checkpoint.txt" in _store_paths(workspace, "scratch_dirs")
