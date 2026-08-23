"""Regression tests for cloud-workspace error precedence."""

# ruff: noqa: D103, EM101, S101, TRY003

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from obstore.store import MemoryStore

import misen.workspaces.cloud as cloud_module
from misen.exceptions import StorageError
from misen.utils.hashing import ResultHash
from misen.workspaces.cloud import CloudWorkspace, ObstoreResultStore


class _WorkerError(RuntimeError):
    pass


def _streaming_workspace(final_error: BaseException) -> CloudWorkspace:
    def stop(_path: Path) -> None:
        raise final_error

    stub = SimpleNamespace(
        _job_log_remote_key=str,
        _start_live_upload=lambda _path, _key: None,
        _stop_live_upload=stop,
    )
    return cast("CloudWorkspace", cast("Any", stub))


def test_job_log_finalization_does_not_replace_worker_failure(tmp_path: Path) -> None:
    workspace = _streaming_workspace(StorageError("upload failed"))

    with pytest.raises(_WorkerError) as raised:
        with CloudWorkspace.streaming_job_log(workspace, tmp_path / "job.log"):
            raise _WorkerError("task failed")

    assert any("upload failed" in note for note in raised.value.__notes__)


def test_job_log_finalization_failure_surfaces_after_success(tmp_path: Path) -> None:
    workspace = _streaming_workspace(StorageError("upload failed"))

    with pytest.raises(StorageError, match="upload failed"):
        with CloudWorkspace.streaming_job_log(workspace, tmp_path / "job.log"):
            pass


class _StuckThread:
    def __init__(self) -> None:
        self.joined = False

    def is_alive(self) -> bool:
        return True

    def join(self, *, timeout: float) -> None:
        del timeout
        self.joined = True


def test_live_uploader_does_not_compact_while_background_thread_is_alive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploader = cloud_module._LiveLogUploader(object(), tmp_path / "job.log", "job.log", 0.01)
    stuck = _StuckThread()
    uploader._thread = cast("Any", stuck)  # noqa: SLF001
    compacted = False

    def compact(_self: object) -> None:
        nonlocal compacted
        compacted = True

    monkeypatch.setattr(cloud_module._LiveLogUploader, "compact", compact)

    with pytest.raises(StorageError, match="did not stop"):
        uploader.stop(final_upload=True)

    assert stuck.joined
    assert not compacted


def test_scratch_sync_does_not_finalize_while_background_thread_is_alive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync = cloud_module._ScratchDirSync(object(), tmp_path / "scratch", "scratch", 0.01)
    stuck = _StuckThread()
    sync._thread = cast("Any", stuck)  # noqa: SLF001
    finalized = False

    def sync_once(_self: object) -> None:
        nonlocal finalized
        finalized = True

    monkeypatch.setattr(cloud_module._ScratchDirSync, "_sync_once", sync_once)

    with pytest.raises(StorageError, match="did not stop"):
        sync.stop(final_upload=True)

    assert stuck.joined
    assert not finalized


class _Stopper:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.called = False

    def stop(self, *, final_upload: bool) -> None:
        assert final_upload
        self.called = True
        if self.error is not None:
            raise self.error


def test_cloud_close_attempts_every_resource_and_retains_failed_handles(tmp_path: Path) -> None:
    first_error = StorageError("first upload failed")
    first = _Stopper(first_error)
    second = _Stopper(StorageError("scratch upload failed"))
    successful = _Stopper()
    failed_path = tmp_path / "failed.log"
    successful_path = tmp_path / "successful.log"
    stub = SimpleNamespace(
        _lifecycle_lock=threading.Lock(),
        _closed=False,
        _live_log_lock=threading.Lock(),
        _live_log_uploaders={failed_path: first, successful_path: successful},
        _scratch_dir_lock=threading.Lock(),
        _scratch_dir_syncs={"failed": second},
        _store=object(),
    )
    workspace = cast("CloudWorkspace", cast("Any", stub))

    with pytest.raises(StorageError, match="first upload failed") as raised:
        CloudWorkspace.close(workspace)

    assert first.called and second.called and successful.called
    assert workspace._live_log_uploaders == {failed_path: first}  # noqa: SLF001
    assert workspace._scratch_dir_syncs == {"failed": second}  # noqa: SLF001
    assert hasattr(workspace, "_store")
    assert any("scratch upload failed" in note for note in raised.value.__notes__)


def test_cloud_result_delete_surfaces_failed_local_eviction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = ResultHash(1)
    results = ObstoreResultStore(MemoryStore(), "results", tmp_path / "cache")
    local = tmp_path / "cache" / key.b32()
    local.mkdir()

    def fail_rmtree(_path: Path) -> None:
        raise OSError("cache eviction failed")

    monkeypatch.setattr(cloud_module.shutil, "rmtree", fail_rmtree)

    with pytest.raises(StorageError, match="cache eviction failed"):
        del results[key]

    assert local.exists()
