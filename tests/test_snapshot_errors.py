"""Exception-contract tests for snapshot creation and environment builds."""
# ruff: noqa: D103, S101, SLF001

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, cast

import pytest

from misen.exceptions import SnapshotError, StorageError
from misen.utils import snapshot as snapshot_mod

if TYPE_CHECKING:
    from pathlib import Path

    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace


class _PayloadWorkspace:
    def __init__(self, temp_dir: Path) -> None:
        self.temp_dir = temp_dir

    def get_temp_dir(self) -> Path:
        return self.temp_dir


class _PayloadWorkUnit:
    def as_payload(self, *, workspace: object, job_id: str) -> bytes:
        del workspace, job_id
        return b"payload"


@pytest.mark.parametrize(
    "cause",
    [
        subprocess.CalledProcessError(2, ["tool"], stderr="invalid lock"),
        FileNotFoundError("tool is missing"),
    ],
)
def test_run_tool_translates_expected_process_failures(
    cause: subprocess.CalledProcessError | FileNotFoundError,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise cause

    monkeypatch.setattr(snapshot_mod.subprocess, "run", fail)

    with pytest.raises(SnapshotError, match="environment build failed") as raised:
        snapshot_mod._run_tool(["tool", "build"], error_msg="environment build failed")

    assert raised.value.__cause__ is cause


def test_run_tool_does_not_mask_unexpected_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    cause = ValueError("programmer error")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise cause

    monkeypatch.setattr(snapshot_mod.subprocess, "run", fail)

    with pytest.raises(ValueError, match="programmer error") as raised:
        snapshot_mod._run_tool(["tool"], error_msg="build failed")

    assert raised.value is cause


def test_invalid_lockfile_is_snapshot_error_with_parse_cause(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("not = [valid")

    with pytest.raises(SnapshotError, match="Invalid uv lockfile") as raised:
        snapshot_mod._local_package_paths(lock)

    assert isinstance(raised.value.__cause__, snapshot_mod.tomllib.TOMLDecodeError)


def test_materialization_io_failure_is_snapshot_error(tmp_path: Path) -> None:
    project_dir = tmp_path / "missing-project"

    with pytest.raises(SnapshotError, match="Could not materialize environments") as raised:
        snapshot_mod._materialize_envs(project_dir, tmp_path / "store")

    assert isinstance(raised.value.__cause__, FileNotFoundError)


def test_uv_resolution_failure_is_snapshot_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cause = RuntimeError("uv is unavailable")

    def fail() -> str:
        raise cause

    snapshot_mod._uv_bin.cache_clear()
    monkeypatch.setattr(snapshot_mod, "find_or_install_uv", fail)
    try:
        with pytest.raises(SnapshotError, match="usable uv executable") as raised:
            snapshot_mod._uv_bin()
    finally:
        snapshot_mod._uv_bin.cache_clear()

    assert raised.value.__cause__ is cause


@pytest.mark.parametrize(
    ("blocked_at", "expected_cause"),
    [("directory", NotADirectoryError), ("payload", IsADirectoryError)],
)
def test_prepare_live_job_translates_payload_io_failures(
    blocked_at: str,
    expected_cause: type[OSError],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(snapshot_mod, "token_base32", lambda _length: "job")
    temp_dir = tmp_path / "temp"
    if blocked_at == "directory":
        temp_dir.write_bytes(b"not a directory")
    else:
        (temp_dir / "live_payloads" / "job.pkl").mkdir(parents=True)

    with pytest.raises(StorageError, match="live job payload") as raised:
        snapshot_mod.prepare_live_job(
            cast("WorkUnit", _PayloadWorkUnit()),
            cast("Workspace", _PayloadWorkspace(temp_dir)),
            cpu_indices=None,
            accelerator_type="cuda",
            accelerator_indices=None,
        )

    assert isinstance(raised.value.__cause__, expected_cause)
