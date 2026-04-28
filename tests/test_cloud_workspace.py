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
from obstore.store import MemoryStore

from misen import WORK_DIR, Task, meta
from misen.exceptions import LockUnavailableError
from misen.utils.hashing import ResolvedTaskHash, ResultHash, TaskHash
from misen.utils.locks import LockLike, ObjectStoreLock
from misen.utils.settings import Settings
from misen.workspace import Workspace
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


@meta(id="cloud_test_writes_workdir", cache=True, exclude={"work_dir"})
def cloud_test_writes_workdir(work_dir: Path, value: str) -> int:
    """Write ``value`` to a fixed file inside the runtime work_dir."""
    (work_dir / "marker.txt").write_text(value)
    return len(value)


@meta(id="cloud_test_reads_workdir", cache=True, exclude={"work_dir"})
def cloud_test_reads_workdir(work_dir: Path) -> str:
    """Return the contents of a checkpoint file restored from durable storage."""
    f = work_dir / "preserved.txt"
    if f.exists():
        return f.read_text()
    return "fresh"


@meta(id="cloud_test_writes_workdir_cleanup", cache=True, cleanup_work_dir=True, exclude={"work_dir"})
def cloud_test_writes_workdir_cleanup(work_dir: Path) -> int:
    """Write to work_dir; the runtime will clean it up after the task succeeds."""
    (work_dir / "to_remove.txt").write_text("bye")
    return 7


@meta(id="cloud_test_writes_workdir_no_cache", cache=False, exclude={"work_dir"})
def cloud_test_writes_workdir_no_cache(work_dir: Path) -> int:
    """Non-cacheable task that writes to its ephemeral work_dir."""
    (work_dir / "ephemeral.txt").write_text("nope")
    return 1


# Module-level events for the during-execution sync test. Tests must reset
# them before use because the module is shared across tests.
_during_exec_can_finish = threading.Event()
_during_exec_observed = threading.Event()


@meta(id="cloud_test_pauses_in_workdir", cache=True, exclude={"work_dir"})
def cloud_test_pauses_in_workdir(work_dir: Path) -> int:
    """Write a checkpoint, pause for the test to inspect the bucket, then continue."""
    (work_dir / "early.txt").write_text("written-early")
    _during_exec_observed.set()
    if not _during_exec_can_finish.wait(timeout=10):
        msg = "test driver never released the during-execution gate"
        raise RuntimeError(msg)
    (work_dir / "late.txt").write_text("written-late")
    return 42


def _workspace(tmp_path, bucket: str) -> _MemoryCloudWorkspace:
    return _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        scratch_dir=str(tmp_path / "scratch"),
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
    """Distinct identities resolve to distinct scratch subdirs under one base."""
    base = tmp_path / "scratch"
    ws_default = _MemoryCloudWorkspace(backend="s3", bucket="b", scratch_dir=str(base))
    ws_prefix = _MemoryCloudWorkspace(backend="s3", bucket="b", prefix="x", scratch_dir=str(base))
    ws_endpoint = _MemoryCloudWorkspace(
        backend="s3", bucket="b", endpoint="https://r2.example", scratch_dir=str(base)
    )
    ws_region = _MemoryCloudWorkspace(
        backend="s3", bucket="b", s3_region="us-west-2", scratch_dir=str(base)
    )

    ids = {ws.workspace_id for ws in (ws_default, ws_prefix, ws_endpoint, ws_region)}
    assert len(ids) == 4
    for ws in (ws_default, ws_prefix, ws_endpoint, ws_region):
        assert ws._scratch.parent == base
        assert ws._scratch.name == ws.workspace_id

    # Same identity => same id => same scratch subdir (safe sharing).
    twin = _MemoryCloudWorkspace(backend="s3", bucket="b", scratch_dir=str(base))
    assert twin.workspace_id == ws_default.workspace_id
    assert twin._scratch == ws_default._scratch


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
            scratch_dir=str(tmp_path / "scratch"),
        )


def test_cloud_workspace_caches_persist_across_instances(tmp_path) -> None:
    """A second workspace with a fresh scratch dir loads results from the bucket."""
    bucket = "test-persist"
    ws_a = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        scratch_dir=str(tmp_path / "scratch-a"),
    )
    task = Task(cloud_test_task_a)
    result = task.result(workspace=ws_a, compute_if_uncached=True, compute_uncached_deps=True)
    assert result == 11
    assert task.is_cached(workspace=ws_a)

    ws_b = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        scratch_dir=str(tmp_path / "scratch-b"),
    )
    assert task.is_cached(workspace=ws_b)
    assert task.result(workspace=ws_b) == 11


def test_cloud_workspace_log_uploaded_and_downloaded(tmp_path) -> None:
    """Log writes upload on close and a different scratch dir downloads them."""
    bucket = "test-logs"
    ws_a = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        scratch_dir=str(tmp_path / "scratch-a"),
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
        scratch_dir=str(tmp_path / "scratch-b"),
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


def test_workspace_auto_resolves_cloud_from_toml(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``Workspace.auto`` resolves the ``cloud`` alias with the new fields."""
    config = tmp_path / "misen.toml"
    config.write_text(
        f"""[workspace]
type = "cloud"
backend = "s3"
bucket = "auto-resolved"
scratch_dir = "{tmp_path / "scratch"}"
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
        scratch_dir=str(tmp_path / "scratch-2"),
    )
    paths = list(other.job_log_iter(work_unit=work_unit))
    assert len(paths) == 1
    assert paths[0].read_text() == "job output"


def test_cloud_workspace_job_log_chunks_are_compacted_on_finalize(tmp_path) -> None:
    """Live job-log chunks are removed once the canonical log is uploaded."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-job-chunk-compaction",
        scratch_dir=str(tmp_path / "scratch"),
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
        scratch_dir=str(tmp_path / "scratch-a"),
    )
    work_unit = _work_unit_for(cloud_test_task_a)
    log_path = workspace.get_job_log(job_id="job-1", work_unit=work_unit)
    remote_key = workspace._job_log_remote_key(log_path)

    other = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        scratch_dir=str(tmp_path / "scratch-b"),
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
        scratch_dir=str(tmp_path / "scratch-a"),
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
        scratch_dir=str(tmp_path / "scratch-b"),
    )

    paths = list(other.job_log_iter(work_unit=work_unit))
    assert len(paths) == 1
    assert paths[0].read_text() == "last complete\n"


def test_cloud_workspace_refresh_does_not_overwrite_active_local_log(tmp_path) -> None:
    """Remote refresh avoids local logs that this workspace is uploading."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-active-log-refresh",
        scratch_dir=str(tmp_path / "scratch"),
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
        scratch_dir=str(tmp_path / "scratch-a"),
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
        scratch_dir=str(tmp_path / "scratch-b"),
    )
    paths = list(ws_b.job_log_iter(work_unit=work_unit))
    assert paths[0].read_text() == "worker output\n"

    # After finalize, the bucket reflects the file's current state.
    ws_a.finalize_job_log(log_path)

    ws_c = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        scratch_dir=str(tmp_path / "scratch-c"),
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
        scratch_dir=str(tmp_path / "scratch-2"),
    )
    paths = list(other.job_log_iter(work_unit=work_unit))
    assert paths[0].read_text() == "content"


def test_cloud_workspace_job_log_iter_filters_by_work_unit(tmp_path) -> None:
    """job_log_iter merges local + remote and respects the work_unit filter."""
    workspace = _workspace(tmp_path, "test-jli-filter")
    wu_a = _work_unit_for(cloud_test_task_a)
    wu_b = _work_unit_for(cloud_test_task_b_for_filter)

    for job_id, wu, body in (("a1", wu_a, "a1"), ("a2", wu_a, "a2"), ("b1", wu_b, "b1")):
        log_path = workspace.get_job_log(job_id=job_id, work_unit=wu)
        with workspace.streaming_job_log(log_path):
            log_path.write_text(body)

    # Fresh scratch dir ensures the merge fetches everything from the bucket.
    other = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-jli-filter",
        scratch_dir=str(tmp_path / "scratch-2"),
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
        scratch_dir=str(tmp_path / "scratch-a"),
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
            scratch_dir=str(tmp_path / "scratch-b"),
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
        scratch_dir=str(tmp_path / "scratch-c"),
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
        scratch_dir=str(tmp_path / "scratch-a"),
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
            scratch_dir=str(tmp_path / "scratch-b"),
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
        scratch_dir=str(tmp_path / "scratch-c"),
    )
    paths = list(ws_c.job_log_iter(work_unit=work_unit))
    assert len(paths) == 1
    assert paths[0].read_text().splitlines() == ["partial output", "more output"]


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
    scratch = tmp_path / "scratch-shared"
    writer_ws = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        scratch_dir=str(scratch),
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

        # Reader in a separate workspace instance with the same scratch:
        # simulates orchestrator-vs-subprocess (LocalExecutor). The
        # orchestrator's ``_live_log_uploaders`` dict does NOT contain the
        # writer path (a different process owns it), so the in-process
        # guard inside ``_refresh_log_local`` cannot save us here.
        reader_ws = _MemoryCloudWorkspace(
            backend="s3",
            bucket=bucket,
            scratch_dir=str(scratch),
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

    # A fresh client (no shared scratch) reads via the compacted .log blob.
    # If the read had clobbered the writer file, compact would have
    # uploaded only the truncated 12-byte chunk content.
    fresh = _MemoryCloudWorkspace(
        backend="s3",
        bucket=bucket,
        scratch_dir=str(tmp_path / "scratch-fresh"),
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
        scratch_dir=str(tmp_path / "scratch"),
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


def test_cloud_workspace_results_iter(tmp_path) -> None:
    """The result store iterates over all stored result hashes."""
    workspace = _workspace(tmp_path, "test-results-iter")
    task = Task(cloud_test_task_a)
    task.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True)
    iter_count = sum(1 for _ in iter(workspace.results.result_store))
    assert iter_count == 1


def _work_dir_remote_paths(workspace: _MemoryCloudWorkspace, task: Task) -> set[str]:
    prefix = workspace._work_dir_remote_prefix(task) + "/"
    return {entry["path"] for batch in obs.list(workspace._store, prefix=prefix) for entry in batch}


def test_cloud_workspace_work_dir_persisted_after_task(tmp_path) -> None:
    """A cacheable task's work_dir contents land in the bucket after completion."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-workdir-persisted",
        scratch_dir=str(tmp_path / "scratch"),
    )
    task = Task(cloud_test_writes_workdir, WORK_DIR, "hello")
    assert task.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True) == 5

    paths = _work_dir_remote_paths(workspace, task)
    assert any(p.endswith("/marker.txt") for p in paths)
    marker_key = next(p for p in paths if p.endswith("/marker.txt"))
    assert bytes(obs.get(workspace._store, marker_key).bytes()) == b"hello"


def test_cloud_workspace_work_dir_restored_from_bucket(tmp_path) -> None:
    """A pre-existing cloud snapshot is restored into the local work_dir before the task runs."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-workdir-restored",
        scratch_dir=str(tmp_path / "scratch"),
    )
    task = Task(cloud_test_reads_workdir, WORK_DIR)

    remote_prefix = workspace._work_dir_remote_prefix(task)
    obs.put(workspace._store, f"{remote_prefix}/preserved.txt", b"from-bucket", mode="overwrite")

    result = task.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True)
    assert result == "from-bucket"


def test_cloud_workspace_work_dir_synced_during_execution(tmp_path) -> None:
    """Files written during a long-running task land in the bucket before completion."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-workdir-during",
        scratch_dir=str(tmp_path / "scratch"),
        work_dir_sync_interval_s=0.05,
    )
    task = Task(cloud_test_pauses_in_workdir, WORK_DIR)
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
        paths = _work_dir_remote_paths(workspace, task)
        assert any(p.endswith("/early.txt") for p in paths)
        assert not any(p.endswith("/late.txt") for p in paths)
    finally:
        _during_exec_can_finish.set()
        runner.join(timeout=10)

    assert not runner_error, runner_error[0]
    paths = _work_dir_remote_paths(workspace, task)
    assert any(p.endswith("/early.txt") for p in paths)
    assert any(p.endswith("/late.txt") for p in paths)


def test_cloud_workspace_work_dir_cleanup_removes_local_and_remote(tmp_path) -> None:
    """``cleanup_work_dir=True`` deletes both the local cache and the bucket prefix."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-workdir-cleanup",
        scratch_dir=str(tmp_path / "scratch"),
    )
    task = Task(cloud_test_writes_workdir_cleanup, WORK_DIR)
    assert task.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True) == 7

    paths = _work_dir_remote_paths(workspace, task)
    assert paths == set()
    local_path = workspace._scratch / "work" / task.resolved_hash(workspace=workspace).b32()
    assert not local_path.exists()


def test_cloud_workspace_work_dir_persists_when_no_cleanup(tmp_path) -> None:
    """Without ``cleanup_work_dir`` the local cache is preserved alongside the bucket copy."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-workdir-no-cleanup",
        scratch_dir=str(tmp_path / "scratch"),
    )
    task = Task(cloud_test_writes_workdir, WORK_DIR, "keep")
    task.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True)

    local_path = workspace._scratch / "work" / task.resolved_hash(workspace=workspace).b32()
    assert (local_path / "marker.txt").read_text() == "keep"
    paths = _work_dir_remote_paths(workspace, task)
    assert any(p.endswith("/marker.txt") for p in paths)


def test_cloud_workspace_remove_work_dir_when_idle(tmp_path) -> None:
    """``remove_work_dir`` works when no sync session is currently active."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-workdir-remove-idle",
        scratch_dir=str(tmp_path / "scratch"),
    )
    task = Task(cloud_test_writes_workdir, WORK_DIR, "data")
    task.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True)

    workspace.remove_work_dir(task=task)
    assert _work_dir_remote_paths(workspace, task) == set()
    local_path = workspace._scratch / "work" / task.resolved_hash(workspace=workspace).b32()
    assert not local_path.exists()


def test_cloud_workspace_remove_work_dir_rejects_non_cacheable(tmp_path) -> None:
    """``remove_work_dir`` refuses non-cacheable tasks."""
    workspace = _workspace(tmp_path, "test-workdir-remove-rejects")
    task = Task(cloud_test_writes_workdir_no_cache, WORK_DIR)
    with pytest.raises(RuntimeError, match="cannot use workspace work_dir"):
        workspace.remove_work_dir(task=task)


def test_cloud_workspace_work_dir_not_synced_for_non_cacheable(tmp_path) -> None:
    """Non-cacheable tasks use ephemeral local work_dirs and never publish to the bucket."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-workdir-non-cache",
        scratch_dir=str(tmp_path / "scratch"),
    )
    task = Task(cloud_test_writes_workdir_no_cache, WORK_DIR)
    task.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True)

    work_dir_paths = {
        entry["path"]
        for batch in obs.list(workspace._store, prefix=workspace._under("work_dirs") + "/")
        for entry in batch
    }
    assert work_dir_paths == set()


def test_cloud_workspace_close_stops_work_dir_syncs(tmp_path) -> None:
    """Closing the workspace tears down any outstanding work_dir sync threads."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-workdir-close",
        scratch_dir=str(tmp_path / "scratch"),
        work_dir_sync_interval_s=0.05,
    )
    task = Task(cloud_test_writes_workdir, WORK_DIR, "data")
    workspace.start_work_dir_sync(task=task)
    assert workspace._work_dir_syncs

    workspace.close()
    assert not workspace._work_dir_syncs


def test_cloud_workspace_work_dir_sync_drops_deleted_files(tmp_path) -> None:
    """Files removed locally during execution are removed from the bucket on the next tick."""
    workspace = _MemoryCloudWorkspace(
        backend="s3",
        bucket="test-workdir-deletion",
        scratch_dir=str(tmp_path / "scratch"),
        work_dir_sync_interval_s=0.05,
    )
    task = Task(cloud_test_writes_workdir, WORK_DIR, "value")
    workspace.start_work_dir_sync(task=task)
    try:
        local_dir = workspace._get_work_dir(task)
        (local_dir / "transient.txt").write_text("temporary")
        time.sleep(0.25)
        paths = _work_dir_remote_paths(workspace, task)
        assert any(p.endswith("/transient.txt") for p in paths)

        (local_dir / "transient.txt").unlink()
        time.sleep(0.25)
        paths = _work_dir_remote_paths(workspace, task)
        assert not any(p.endswith("/transient.txt") for p in paths)
    finally:
        workspace.finalize_work_dir(task=task)
