"""Tests for the optional conda-environment path of ``LocalSnapshot``."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from misen.utils.snapshot import LocalSnapshot

FIXTURES = Path(__file__).parent / "fixtures" / "conda_snapshot"
ZLIB_XZ_LOCK = FIXTURES / "pixi.lock"
ZLIB_XZ_MANIFEST = FIXTURES / "pixi.toml"

pytestmark = pytest.mark.skipif(shutil.which("pixi") is None, reason="pixi CLI not installed")


def _stage_uv_project(root: Path) -> None:
    """Write a minimal pyproject.toml + uv.lock under ``root`` for uv sync."""
    (root / "pyproject.toml").write_text(
        '[project]\nname = "conda-snap-test"\nversion = "0.0.0"\nrequires-python = ">=3.11"\n'
    )
    (root / "README.md").write_text("")
    subprocess.run(["uv", "lock"], cwd=root, check=True, capture_output=True)  # noqa: S603, S607


def _write_minimal_pixi_manifest(root: Path) -> None:
    """Minimal pixi.toml so pixi accepts the manifest; content doesn't matter
    for rejection tests because the lockfile is invalid before pixi sees it."""
    (root / "pixi.toml").write_text(
        "[workspace]\n"
        'name = "rejection-fixture"\n'
        'channels = ["conda-forge"]\n'
        'platforms = ["osx-arm64", "linux-64", "osx-64", "linux-aarch64"]\n'
        "[dependencies]\n"
    )


# ---------- error-path tests via LocalSnapshot ----------


def test_local_snapshot_rejects_pypi_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    with pytest.raises(RuntimeError, match="pypi dependencies"):
        LocalSnapshot(snapshots_dir=tmp_path / "snapshots")


def test_local_snapshot_rejects_missing_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Lockfile without adjacent pixi.toml -> clear error."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    _stage_uv_project(project_root)
    shutil.copy(ZLIB_XZ_LOCK, project_root / "pixi.lock")
    monkeypatch.chdir(project_root)

    with pytest.raises(RuntimeError, match="no pixi.toml"):
        LocalSnapshot(snapshots_dir=tmp_path / "snapshots")


# ---------- end-to-end (installs real tiny env) ----------


def test_local_snapshot_wraps_argv_in_pixi_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: install zlib+xz, confirm prepare_job wraps argv in ``pixi run``
    and that running the wrapped argv activates the snapshot's conda env."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    _stage_uv_project(project_root)
    shutil.copy(ZLIB_XZ_MANIFEST, project_root / "pixi.toml")
    shutil.copy(ZLIB_XZ_LOCK, project_root / "pixi.lock")
    monkeypatch.chdir(project_root)

    snapshots_dir = tmp_path / "snapshots"
    snapshot = LocalSnapshot(snapshots_dir=snapshots_dir)
    try:
        # Snapshot has a staged manifest and a cached pixi bin.
        assert snapshot.conda_manifest_path is not None
        assert snapshot.conda_manifest_path.is_file()
        assert snapshot.conda_manifest_path.is_relative_to(snapshot.snapshot_dir)
        assert snapshot.pixi_bin is not None

        # pixi install ran already, so the env exists under snapshot_dir.
        prefix_dir = snapshot.snapshot_dir / ".pixi" / "envs" / "default"
        assert prefix_dir.is_dir()
        xz_bin = prefix_dir / "bin" / "xz"
        assert xz_bin.is_file() and os.access(xz_bin, os.X_OK)
        assert list((prefix_dir / "lib").glob("libz.*"))

        # Run a tiny command through the same wrapper prepare_job would build
        # and confirm CONDA_PREFIX + PATH land correctly.
        wrapped = [
            snapshot.pixi_bin,
            "run",
            "--no-progress",
            "--color",
            "never",
            "--frozen",
            "--manifest-path",
            str(snapshot.conda_manifest_path),
            "-x",
            "--",
            "/usr/bin/env",
            "bash",
            "-c",
            'printf "%s\\n%s\\n" "$CONDA_PREFIX" "$PATH"',
        ]
        result = subprocess.run(wrapped, capture_output=True, text=True, check=True)  # noqa: S603
        conda_prefix, path_value, *_ = result.stdout.splitlines()
        assert Path(conda_prefix).resolve() == prefix_dir.resolve()
        assert path_value.split(":")[0] == str(prefix_dir / "bin")
    finally:
        snapshot.cleanup()

    assert not snapshot.snapshot_dir.exists()


def test_local_snapshot_no_conda_when_pixi_lock_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """LocalSnapshot without pixi.lock leaves ``conda_manifest_path`` as None."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    _stage_uv_project(project_root)
    monkeypatch.chdir(project_root)

    snapshot = LocalSnapshot(snapshots_dir=tmp_path / "snapshots")
    try:
        assert snapshot.conda_manifest_path is None
    finally:
        snapshot.cleanup()
