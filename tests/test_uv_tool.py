"""Standalone uv discovery and managed-install fallback."""
# ruff: noqa: D103, S101

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from misen.utils.uv_tool import UV_AUTO_INSTALL_ENV, UV_BIN_ENV, UV_VERSION, find_or_install_uv


def _write_uv(path: Path, version: str = UV_VERSION) -> Path:
    path.write_text(f"#!/bin/sh\nprintf 'uv {version}\\n'\n")
    path.chmod(0o755)
    return path


def _write_downloader(path: Path, calls: Path, version: str = UV_VERSION) -> None:
    path.write_text(
        "#!/bin/sh\n"
        f"printf 'download\\n' >> {calls!s}\n"
        "cat <<'INSTALLER'\n"
        "#!/bin/sh\n"
        'mkdir -p "$UV_UNMANAGED_INSTALL"\n'
        "cat > \"$UV_UNMANAGED_INSTALL/uv\" <<'UV'\n"
        "#!/bin/sh\n"
        f"printf 'uv {version}\\n'\n"
        "UV\n"
        'chmod 755 "$UV_UNMANAGED_INSTALL/uv"\n'
        "INSTALLER\n"
    )
    path.chmod(0o755)


def _clear_resolution() -> None:
    find_or_install_uv.cache_clear()


def test_configured_uv_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    uv = _write_uv(tmp_path / "configured-uv")
    monkeypatch.setenv(UV_BIN_ENV, str(uv))
    _clear_resolution()
    try:
        assert find_or_install_uv() == str(uv)
    finally:
        _clear_resolution()


def test_invalid_configured_uv_falls_back_to_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    uv = _write_uv(tmp_path / "uv")
    monkeypatch.setenv(UV_BIN_ENV, str(tmp_path / "missing"))
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    _clear_resolution()
    try:
        assert find_or_install_uv() == str(uv)
    finally:
        _clear_resolution()


def test_managed_uv_is_installed_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    _write_uv(tools / "uv", "0.1.0")
    calls = tmp_path / "downloads"
    _write_downloader(tools / "curl", calls)
    monkeypatch.delenv(UV_BIN_ENV, raising=False)
    monkeypatch.setenv("PATH", f"{tools}:/usr/bin:/bin")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    _clear_resolution()
    try:
        first = find_or_install_uv()
        _clear_resolution()
        second = find_or_install_uv()
    finally:
        _clear_resolution()

    assert first == second
    assert Path(first).is_relative_to(tmp_path / "data" / "misen" / "tools" / "uv" / UV_VERSION)
    assert subprocess.run([first, "--version"], check=True, capture_output=True, text=True).stdout == (  # noqa: S603
        f"uv {UV_VERSION}\n"
    )
    assert calls.read_text().splitlines() == ["download"]


def test_managed_install_rejects_wrong_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    _write_uv(tools / "uv", "0.1.0")
    _write_downloader(tools / "curl", tmp_path / "downloads", "0.1.0")
    data_home = tmp_path / "data"
    monkeypatch.delenv(UV_BIN_ENV, raising=False)
    monkeypatch.setenv("PATH", f"{tools}:/usr/bin:/bin")
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    _clear_resolution()
    try:
        with pytest.raises(RuntimeError, match="uv installer failed"):
            find_or_install_uv()
    finally:
        _clear_resolution()

    assert not any(path.is_file() for path in data_home.rglob("uv"))


def test_automatic_install_can_be_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    _write_uv(tools / "uv", "0.1.0")
    bash = shutil.which("bash")
    assert bash is not None
    (tools / "bash").symlink_to(bash)
    monkeypatch.delenv(UV_BIN_ENV, raising=False)
    monkeypatch.setenv(UV_AUTO_INSTALL_ENV, "0")
    monkeypatch.setenv("PATH", f"{tools}:/usr/bin:/bin")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    _clear_resolution()
    try:
        with pytest.raises(RuntimeError, match="uv >="):
            find_or_install_uv()
    finally:
        _clear_resolution()
