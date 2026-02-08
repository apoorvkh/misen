"""Utilities to capture execution environment snapshots."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import uuid
from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import uv

if TYPE_CHECKING:
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace


class Snapshot(ABC):
    """Abstract environment snapshot used by executors."""

    __slots__ = ()

    @abstractmethod
    def command(self, work_unit: WorkUnit, workspace: Workspace) -> list[str]:
        """Build the command used to execute a work unit."""

    @abstractmethod
    def command_env(self) -> dict[str, str]:
        """Build environment variable overrides used for execution."""


class LocalSnapshot(Snapshot):
    """Environment snapshot materialized on local disk for task execution."""

    __slots__ = ("env_files", "snapshot_dir", "uv_bin", "venv_dir")

    def __init__(self, snapshots_dir: Path) -> None:
        """Create a fresh snapshot directory and materialize environment state."""
        self.uv_bin = uv.find_uv_bin()
        self.snapshot_dir = snapshots_dir / f"{uuid.uuid4().hex}"
        self.snapshot_dir.mkdir(parents=True)
        self.venv_dir = self._snapshot_venv(self.snapshot_dir / ".venv")
        self.env_files = self._snapshot_env_files(self.snapshot_dir)

    def command(self, work_unit: WorkUnit, workspace: Workspace) -> list[str]:
        """Build a uv command to execute a serialized work-unit payload."""
        payload_dir = self.snapshot_dir / "payloads"
        payload_dir.mkdir(parents=True, exist_ok=True)
        payload_path = payload_dir / f"{uuid.uuid4().hex}.pkl"
        payload_path.write_bytes(work_unit.as_payload(workspace=workspace))
        return [
            self.uv_bin,
            "run",
            "--no-project",
            *chain.from_iterable(("--env-file", str(path)) for path in self.env_files),
            "-m",
            "misen.utils.execute",
            "--payload",
            str(payload_path),
        ]

    def command_env(self) -> dict[str, str]:
        """Build process environment overrides for execution."""
        return {"VIRTUAL_ENV": str(self.venv_dir)}

    def _snapshot_venv(self, venv_dir: Path) -> Path:
        """Install a frozen snapshot of current package and dependencies into a virtual env."""
        env = os.environ.copy() | {"UV_PROJECT_ENVIRONMENT": str(venv_dir)}

        # `uv sync --no-editable` would use cached, but outdated, copies of (local) workspace members
        # instead we can install (1) non-workspace deps with caching then (2) workspace members without caching
        try:
            subprocess.run(  # noqa: S603
                [self.uv_bin, "sync", "--no-install-workspace"], check=True, capture_output=True, text=True, env=env
            )
            subprocess.run(  # noqa: S603
                [self.uv_bin, "sync", "--no-editable", "--no-cache"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as e:
            msg = f"Virtual environment creation failed: {(e.stderr or e.stdout or '').strip()}"
            raise RuntimeError(msg) from None

        return venv_dir

    def _snapshot_env_files(self, snapshot_dir: Path) -> list[Path]:
        """Make a frozen copy of `.env` and `.env.local` files."""
        env_files = []
        for f in (".env", ".env.local"):
            src = Path.cwd() / f
            dst = snapshot_dir / f
            with contextlib.suppress(FileNotFoundError):
                shutil.copy(src, dst)
                env_files.append(dst)
                # owner-only permissions for .env.local (may contain secrets)
                if f == ".env.local":
                    with contextlib.suppress(OSError):
                        dst.chmod(0o600)
        return env_files
