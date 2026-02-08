"""Utilities to capture and replicate current virtual environment."""

import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import uv


def snapshot_venv(venv_dir: Path) -> Path:
    """Install a frozen snapshot of current package and dependencies (locked if possible) into an virtual env."""
    uv_bin = uv.find_uv_bin()
    env = os.environ.copy() | {"UV_PROJECT_ENVIRONMENT": str(venv_dir)}

    # `uv sync --no-editable` would use cached, but outdated, copies of (local) workspace members
    # instead we can install (1) non-workspace deps with caching then (2) workspace members without caching
    try:
        subprocess.run([uv_bin, "sync", "--no-install-workspace"], check=True, capture_output=True, text=True, env=env)  # noqa: S603
        subprocess.run(  # noqa: S603
            [uv_bin, "sync", "--no-editable", "--no-cache"], check=True, capture_output=True, text=True, env=env
        )
    except subprocess.CalledProcessError as e:
        msg = f"Virtual environment creation failed: {(e.stderr or e.stdout or '').strip()}"
        raise RuntimeError(msg) from None

    return venv_dir


def snapshot_env_files(env_dir: Path) -> list[Path]:
    """Make a frozen copy of .env and .env.local files."""
    env_files = []
    for f in (".env", ".env.local"):
        src = Path.cwd() / f
        dst = env_dir / f
        with contextlib.suppress(FileNotFoundError):
            shutil.copy(src, dst)
            env_files.append(dst)
            # owner-only permissions for .env.local (may contain secrets)
            if f == ".env.local":
                with contextlib.suppress(OSError):
                    dst.chmod(0o600)
    return env_files
