"""Tests for the optional conda-environment path of ``ProjectSnapshot``."""
# ruff: noqa: D103, S101

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from misen.exceptions import SnapshotError
from misen.utils.snapshot import ProjectSnapshot
from misen.workspaces.disk import DiskWorkspace

FIXTURES = Path(__file__).parent / "fixtures" / "conda_snapshot"
ZLIB_XZ_LOCK = FIXTURES / "pixi.lock"
ZLIB_XZ_MANIFEST = FIXTURES / "pixi.toml"

pytestmark = pytest.mark.skipif(shutil.which("pixi") is None, reason="pixi CLI not installed")


def _stage_uv_project(root: Path) -> None:
    """Write a minimal pyproject.toml + uv.lock under ``root``."""
    (root / "pyproject.toml").write_text(
        '[project]\nname = "conda-snap-test"\nversion = "0.0.0"\nrequires-python = ">=3.11"\n'
    )
    (root / "README.md").write_text("")
    subprocess.run(["uv", "lock"], cwd=root, check=True, capture_output=True)  # noqa: S607


def _write_minimal_pixi_manifest(root: Path) -> None:
    """Write a minimal pixi.toml so pixi accepts the manifest.

    Content doesn't matter for rejection tests: the lockfile is invalid
    before pixi ever reads it.
    """
    (root / "pixi.toml").write_text(
        "[workspace]\n"
        'name = "rejection-fixture"\n'
        'channels = ["conda-forge"]\n'
        'platforms = ["osx-arm64", "linux-64", "osx-64", "linux-aarch64"]\n'
        "[dependencies]\n"
    )


def _workspace(tmp_path: Path) -> DiskWorkspace:
    return DiskWorkspace(directory=str(tmp_path / ".misen"))


# ---------- error-path tests via ProjectSnapshot staging ----------


def test_project_snapshot_rejects_pypi_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _stage_uv_project(project_root)
    _write_minimal_pixi_manifest(project_root)
    (project_root / "pixi.lock").write_text(
        "version: 6\n"
        "environments:\n"
        "  default:\n"
        "    packages:\n"
        "      osx-arm64:\n"
        "      - pypi: https://example.com/requests-2.31.0-py3-none-any.whl\n"
    )
    monkeypatch.chdir(project_root)

    with pytest.raises(SnapshotError, match="pypi dependencies"):
        ProjectSnapshot(workspace=_workspace(tmp_path))


def test_project_snapshot_rejects_missing_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Lockfile without adjacent pixi.toml -> clear error."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    _stage_uv_project(project_root)
    shutil.copy(ZLIB_XZ_LOCK, project_root / "pixi.lock")
    monkeypatch.chdir(project_root)

    with pytest.raises(SnapshotError, match="no pixi"):
        ProjectSnapshot(workspace=_workspace(tmp_path))


# ---------- end-to-end (installs real tiny env) ----------


def test_prewarmed_snapshot_wraps_argv_in_pixi_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Install zlib+xz for real; the conda entry lands in the env store.

    Also confirms the pixi wrapper activates the snapshot's conda env.
    """
    project_root = tmp_path / "project"
    project_root.mkdir()
    _stage_uv_project(project_root)
    shutil.copy(ZLIB_XZ_MANIFEST, project_root / "pixi.toml")
    shutil.copy(ZLIB_XZ_LOCK, project_root / "pixi.lock")
    monkeypatch.chdir(project_root)

    workspace = _workspace(tmp_path)
    store_root = tmp_path / "env-store"
    snapshot = ProjectSnapshot(workspace=workspace, env_store_dir=str(store_root), prewarm=True)
    # The staged snapshot carries the manifests; the conda env is a
    # store entry beside the staged copy of the manifest.
    assert snapshot.project_dir is not None
    assert (snapshot.project_dir / "pixi.lock").is_file()
    assert snapshot.pixi_bin is not None
    assert snapshot.prewarmed is not None
    manifest_path = snapshot.prewarmed.conda_manifest_path
    assert manifest_path is not None
    assert manifest_path.is_file()
    assert manifest_path.is_relative_to(store_root / "conda-envs")

    # pixi install ran already, so the env exists beside the staged manifest.
    prefix_dir = manifest_path.parent / ".pixi" / "envs" / "default"
    assert prefix_dir.is_dir()
    xz_bin = prefix_dir / "bin" / "xz"
    assert xz_bin.is_file()
    assert os.access(xz_bin, os.X_OK)
    assert list((prefix_dir / "lib").glob("libz.*"))

    # Run a tiny command through the same wrapper prepare_job builds
    # and confirm CONDA_PREFIX + PATH land correctly.
    wrapped = [
        snapshot.prewarmed.pixi_bin,
        "run",
        "--no-progress",
        "--color",
        "never",
        "--frozen",
        "--manifest-path",
        str(manifest_path),
        "--",
        "/usr/bin/env",
        "bash",
        "-c",
        'printf "%s\\n%s\\n%s\\n%s\\n" "$CONDA_PREFIX" "$PATH" "$OMP_NUM_THREADS" "$CUDA_VISIBLE_DEVICES"',
    ]
    launch_env = os.environ.copy()
    launch_env["OMP_NUM_THREADS"] = "2"
    launch_env["CUDA_VISIBLE_DEVICES"] = "1"
    result = subprocess.run(wrapped, env=launch_env, capture_output=True, text=True, check=True)  # noqa: S603
    conda_prefix, path_value, thread_count, visible_devices, *_ = result.stdout.splitlines()
    assert Path(conda_prefix).resolve() == prefix_dir.resolve()
    assert Path(path_value.split(":")[0]).resolve() == (prefix_dir / "bin").resolve()
    assert thread_count == "2"
    assert visible_devices == "1"

    # A second snapshot reuses the conda env entry instead of reinstalling.
    second = ProjectSnapshot(workspace=workspace, env_store_dir=str(store_root), prewarm=True)
    assert second.prewarmed is not None
    assert second.prewarmed.conda_manifest_path == manifest_path

    assert workspace.fetch_snapshot(snapshot.snapshot_key).is_dir()
    assert prefix_dir.is_dir()
    key = manifest_path.parent.name
    assert (store_root / "conda-envs" / f"{key}.complete").is_file()


def test_project_snapshot_no_conda_when_pixi_lock_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ProjectSnapshot without pixi.lock stages no manifests and no conda env."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    _stage_uv_project(project_root)
    monkeypatch.chdir(project_root)

    snapshot = ProjectSnapshot(workspace=_workspace(tmp_path), env_store_dir=str(tmp_path / "env-store"), prewarm=True)
    assert snapshot.project_dir is not None
    assert not (snapshot.project_dir / "pixi.lock").exists()
    assert snapshot.pixi_bin is None
    assert snapshot.prewarmed is not None
    assert snapshot.prewarmed.conda_manifest_path is None
