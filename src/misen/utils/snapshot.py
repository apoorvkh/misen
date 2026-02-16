"""Execution environment snapshots used by executors.

Snapshots capture enough environment state to run work units reproducibly in
subprocesses or on remote schedulers:

- isolated virtual environment
- copied env files
- serialized callable payloads
"""

from __future__ import annotations

import base64
import contextlib
import os
import secrets
import shutil
import subprocess
from abc import ABC, abstractmethod
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import uv
from dotenv import load_dotenv

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from misen.utils.assigned_resources import AssignedResources
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = ["LocalSnapshot", "NullSnapshot", "Snapshot", "apply_env_files_temporarily"]


class Snapshot(ABC):
    """Abstract environment snapshot used by executors."""

    __slots__ = ()

    @abstractmethod
    def prepare_job(
        self,
        work_unit: WorkUnit,
        workspace: Workspace,
        assigned_resources_getter: Callable[[], AssignedResources | None] = lambda: None,
    ) -> tuple[str, list[str], Mapping[str, str]]:
        """Prepare command and environment for one work unit.

        Args:
            work_unit: Work unit to execute.
            workspace: Workspace for payload/log paths.
            assigned_resources_getter: Callable returning runtime resources for
                sentinel injection.

        Returns:
            Tuple ``(job_id, argv, env_overrides)``.
        """


class NullSnapshot(Snapshot):
    """Snapshot placeholder for executors that never dispatch subprocess jobs."""

    __slots__ = ()

    def prepare_job(
        self,
        work_unit: WorkUnit,
        workspace: Workspace,
        assigned_resources_getter: Callable[[], AssignedResources | None] = lambda: None,
    ) -> tuple[str, list[str], Mapping[str, str]]:
        """Raise because null snapshots cannot prepare external commands."""
        _ = work_unit, workspace, assigned_resources_getter
        msg = "NullSnapshot cannot prepare external job commands."
        raise RuntimeError(msg)


class LocalSnapshot(Snapshot):
    """Environment snapshot materialized locally for task execution."""

    __slots__ = ("env_files", "snapshot_dir", "uv_bin", "venv_dir")

    def __init__(self, snapshots_dir: Path) -> None:
        """Create snapshot directory and materialize environment state.

        Args:
            snapshots_dir: Parent directory where snapshots are stored.
        """
        self.uv_bin = uv.find_uv_bin()
        self.snapshot_dir = snapshots_dir / f"{_token_base32(6)}"
        self.snapshot_dir.mkdir(parents=True)
        self.venv_dir = self._snapshot_venv(self.snapshot_dir / ".venv")
        self.env_files = self._snapshot_env_files(self.snapshot_dir)

    def prepare_job(
        self,
        work_unit: WorkUnit,
        workspace: Workspace,
        assigned_resources_getter: Callable[[], AssignedResources | None] = lambda: None,
    ) -> tuple[str, list[str], Mapping[str, str]]:
        """Prepare command/env overrides to execute serialized payload.

        Args:
            work_unit: Work unit to execute.
            workspace: Workspace for payload/log paths.
            assigned_resources_getter: Callable returning runtime resources.

        Returns:
            Tuple ``(job_id, argv, env_overrides)``.
        """
        job_id = _token_base32(6)
        payload_dir = self.snapshot_dir / "payloads"
        payload_dir.mkdir(parents=True, exist_ok=True)
        payload_path = payload_dir / f"{_token_base32(6)}.pkl"
        payload_path.write_bytes(
            work_unit.as_payload(
                workspace=workspace, job_id=job_id, assigned_resources_getter=assigned_resources_getter
            )
        )
        argv: list[str] = [
            self.uv_bin,
            "run",
            "--no-project",
            *chain.from_iterable(("--env-file", str(path)) for path in self.env_files),
            "-m",
            "misen.utils.execute",
            "--payload",
            str(payload_path),
        ]
        env_overrides: dict[str, str] = {"VIRTUAL_ENV": str(self.venv_dir)}
        return job_id, argv, env_overrides

    def _snapshot_venv(self, venv_dir: Path) -> Path:
        """Install a frozen dependency snapshot into virtual environment.

        Args:
            venv_dir: Target virtual environment directory.

        Returns:
            ``venv_dir``.

        Raises:
            RuntimeError: If ``uv sync`` fails.
        """
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
        """Copy supported env files into snapshot directory.

        Args:
            snapshot_dir: Snapshot root directory.

        Returns:
            List of copied env-file paths.
        """
        env_files = []
        for src in _discover_env_files():
            dst = snapshot_dir / src.name
            shutil.copy(src, dst)
            env_files.append(dst)
            # owner-only permissions for .env.local (may contain secrets)
            if src.name == ".env.local":
                with contextlib.suppress(OSError):
                    dst.chmod(0o600)
        return env_files


@contextmanager
def apply_env_files_temporarily() -> Iterator[None]:
    """Temporarily load environment variables from dotenv files.

    Later files override earlier ones. Modified keys are restored after exiting
    the context.

    Args:
        env_files: Optional explicit env-file list. Defaults to
            :func:`discover_env_files` for the current working directory.
    """
    initial_environ = os.environ.copy()
    for f in _discover_env_files():
        load_dotenv(f, override=True)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(initial_environ)


_ENV_FILENAMES = (".env", ".env.local")


def _discover_env_files(cwd: Path | None = None) -> list[Path]:
    """Return existing env files in precedence order.

    Args:
        cwd: Optional directory to search. Defaults to ``Path.cwd()``.

    Returns:
        Existing files among ``.env`` then ``.env.local``.
    """
    root = cwd or Path.cwd()
    return [path for path in (root / name for name in _ENV_FILENAMES) if path.exists()]


def _token_base32(nbytes: int) -> str:
    """Return URL/file-safe random base32 token.

    Args:
        nbytes: Number of random bytes before encoding.

    Returns:
        Base32 token without padding.
    """
    return base64.b32encode(secrets.token_bytes(nbytes)).decode("ascii").rstrip("=")
