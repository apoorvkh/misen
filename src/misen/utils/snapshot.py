"""Utilities to capture and replicate current virtual environment."""

import os
import subprocess
import tempfile
from pathlib import Path

import uv


def snapshot_to_venv(parent_dir: Path) -> Path:
    """Install a frozen snapshot of current package and dependencies (locked if possible) into an virtual env."""
    uv_bin = uv.find_uv_bin()
    parent_dir.mkdir(parents=True, exist_ok=True)
    venv_dir = Path(tempfile.mkdtemp(dir=parent_dir))
    env = os.environ.copy() | {"UV_PROJECT_ENVIRONMENT": str(venv_dir)}

    # `uv sync --no-editable` would use cached, but outdated, copies of local dependencies
    # instead we can (1) install non-local deps, then (2) refresh local deps in cache and install those
    install_deps = [uv_bin, "sync", "--no-install-local"]
    install_local = [uv_bin, "sync", "--inexact", "--only-install-local", "--no-editable", "--refresh"]

    try:
        subprocess.run(install_deps, check=True, capture_output=True, text=True, env=env)  # noqa: S603
        subprocess.run(install_local, check=True, capture_output=True, text=True, env=env)  # noqa: S603
    except subprocess.CalledProcessError as e:
        msg = f"Virtual environment creation failed: {(e.stderr or e.stdout or '').strip()}"
        raise RuntimeError(msg) from None
    return venv_dir
