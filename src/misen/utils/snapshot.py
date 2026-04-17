"""Execution environment snapshots used by executors.

Snapshots capture enough environment state to run work units reproducibly in
subprocesses or on remote schedulers:

- isolated virtual environment (uv)
- optional conda prefix (installed and activated via the ``pixi`` CLI from
  ``pixi.lock`` + ``pixi.toml``)
- copied env files
- serialized callable payloads and assigned-resource getters
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

import cloudpickle
import uv
from dotenv import load_dotenv

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from misen.task_metadata import GpuRuntime
    from misen.utils.assigned_resources import AssignedResources, AssignedResourcesPerNode
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = ["LocalSnapshot", "NullSnapshot", "Snapshot", "apply_env_files_temporarily"]


class Snapshot(ABC):
    """Abstract environment snapshot used by executors."""

    __slots__ = ()

    @abstractmethod
    def cleanup(self) -> None:
        """Remove snapshot artifacts from disk."""

    @abstractmethod
    def prepare_job(
        self,
        work_unit: WorkUnit,
        workspace: Workspace,
        assigned_resources_getter: Callable[[], AssignedResources | AssignedResourcesPerNode | None],
        gpu_runtime: GpuRuntime,
    ) -> tuple[str, list[str], dict[str, str]]:
        """Prepare command and environment for one work unit.

        Args:
            work_unit: Work unit to execute.
            workspace: Workspace for payload/log paths.
            assigned_resources_getter: Callable returning runtime resources for
                sentinel injection and worker CPU/GPU binding.
            gpu_runtime: Runtime environment for GPU resources.

        Returns:
            Tuple ``(job_id, argv, env_overrides)``.
        """


class NullSnapshot(Snapshot):
    """Snapshot placeholder for executors that never dispatch subprocess jobs."""

    __slots__ = ()

    def cleanup(self) -> None:
        """No-op for null snapshots."""

    def prepare_job(
        self,
        work_unit: WorkUnit,
        workspace: Workspace,
        assigned_resources_getter: Callable[[], AssignedResources | AssignedResourcesPerNode | None],
        gpu_runtime: GpuRuntime,
    ) -> tuple[str, list[str], dict[str, str]]:
        """Null snapshots don't prepare external commands."""
        _ = work_unit, workspace, assigned_resources_getter, gpu_runtime
        job_id = token_base32(6)
        return job_id, [], {}


class LocalSnapshot(Snapshot):
    """Environment snapshot materialized locally for task execution.

    Always contains a uv-built virtual environment and copied env files.
    If a ``pixi.lock`` + ``pixi.toml`` pair sits next to the caller's
    CWD, a parallel conda env is installed under the same snapshot
    directory via ``pixi install --frozen``. Jobs are then wrapped in
    ``pixi run --frozen -x -- <uv run ...>`` so activation (``CONDA_PREFIX``,
    ``PATH``, ``LD_LIBRARY_PATH``, plus anything ``activate.d`` scripts
    inject like ``CUDA_HOME``) happens at job spawn. The conda prefix
    supplies native / system libraries while Python and every PyPI
    package stay in the python env.
    """

    __slots__ = (
        "conda_manifest_path",
        "env_files",
        "payload_dir",
        "pixi_bin",
        "python_env_dir",
        "snapshot_dir",
        "uv_bin",
    )

    def __init__(self, snapshots_dir: Path) -> None:
        """Create snapshot directory and materialize environment state.

        Args:
            snapshots_dir: Parent directory where snapshots are stored.
        """
        self.snapshot_dir = snapshots_dir / f"{token_base32(6)}"
        self.snapshot_dir.mkdir(parents=True)

        self.payload_dir = self.snapshot_dir / "payloads"
        self.payload_dir.mkdir(exist_ok=True)

        self.uv_bin = uv.find_uv_bin()
        self.python_env_dir = self._snapshot_python_env(self.snapshot_dir / "python-env")
        self.pixi_bin = None
        self.conda_manifest_path = self._snapshot_conda(self.snapshot_dir)
        self.env_files = self._snapshot_env_files(self.snapshot_dir)

    def cleanup(self) -> None:
        """Remove snapshot directory tree from disk."""
        shutil.rmtree(self.snapshot_dir, ignore_errors=True)

    def prepare_job(
        self,
        work_unit: WorkUnit,
        workspace: Workspace,
        assigned_resources_getter: Callable[[], AssignedResources | AssignedResourcesPerNode | None],
        gpu_runtime: GpuRuntime,
    ) -> tuple[str, list[str], dict[str, str]]:
        """Prepare command/env overrides to execute serialized payload.

        Args:
            work_unit: Work unit to execute.
            workspace: Workspace for payload/log paths.
            assigned_resources_getter: Callable returning runtime resources for
                task sentinel injection and worker binding.
            gpu_runtime: Runtime environment for GPU resources.

        Returns:
            Tuple ``(job_id, argv, env_overrides)``.
        """
        job_id = token_base32(6)

        argv = []

        # When a conda env is present, wrap argv so pixi activates it at job
        # spawn. ``-x`` forces executable mode (no pixi-task lookup); ``--``
        # stops pixi from parsing our command's flags.
        if self.conda_manifest_path is not None and self.pixi_bin is not None:
            argv += [
                self.pixi_bin,
                "run",
                "--no-progress",
                "--color",
                "never",
                "--frozen",
                "--manifest-path",
                str(self.conda_manifest_path),
                "-x",
                "--",
            ]

        payload_path = self.payload_dir / f"{token_base32(6)}.pkl"
        payload_path.write_bytes(work_unit.as_payload(workspace=workspace, job_id=job_id))
        encoded_getter = _encode_cli_blob(cloudpickle.dumps(assigned_resources_getter))

        argv += [
            self.uv_bin,
            "run",
            "--no-project",
            *chain.from_iterable(("--env-file", str(path)) for path in self.env_files),
            "-m",
            "misen.utils.execute",
            "--payload",
            str(payload_path),
            "--assigned-resources-getter",
            encoded_getter,
            "--gpu-runtime",
            gpu_runtime,
        ]

        env_overrides: dict[str, str] = {"VIRTUAL_ENV": str(self.python_env_dir)}

        return job_id, argv, env_overrides

    def _snapshot_python_env(self, python_env_dir: Path) -> Path:
        """Install a frozen dependency snapshot into virtual environment.

        Args:
            python_env_dir: Target virtual environment directory.

        Returns:
            ``python_env_dir``.

        Raises:
            RuntimeError: If ``uv sync`` fails.
        """
        env = os.environ.copy() | {"UV_PROJECT_ENVIRONMENT": str(python_env_dir)}

        # Use a two-step sync to avoid stale cached editable installs:
        # 1) install non-workspace dependencies (cacheable)
        # 2) install workspace members non-editably without cache
        try:
            subprocess.run(  # noqa: S603
                [self.uv_bin, "sync", "--no-install-workspace"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
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

        return python_env_dir

    def _snapshot_conda(self, snapshot_dir: Path) -> Path | None:
        """Install an optional conda env from ``pixi.lock`` via the pixi CLI.

        Stages ``pixi.toml`` + ``pixi.lock`` from CWD into ``snapshot_dir``
        and pre-installs the env via ``pixi install --frozen``. Pixi
        writes the env into ``snapshot_dir/.pixi/envs/default``, so the
        whole snapshot (python env + conda env) is still a single-
        ``rmtree`` on cleanup. Activation is deferred to job-spawn time:
        :meth:`prepare_job` wraps ``argv`` in ``pixi run --frozen -x -- ...``
        so each job starts inside a fresh activation that runs
        ``etc/conda/activate.d/*.sh`` against live env. Any conda
        ``python`` record is installed as-is; the python env still owns
        the interpreter at runtime because ``uv run`` prepends
        ``<python_env_dir>/bin`` ahead of the conda prefix on ``PATH``.

        Args:
            snapshot_dir: Snapshot root directory.

        Returns:
            Path to the staged ``pixi.toml`` (pixi's manifest-path flag
            consumes this). ``None`` when no ``pixi.lock`` is present in
            CWD.

        Raises:
            RuntimeError: If ``pixi.lock`` has no adjacent ``pixi.toml``,
                the lockfile references PyPI packages, the ``pixi`` CLI is
                missing, or ``pixi install`` fails.
        """
        lock_path = Path.cwd() / "pixi.lock"
        if not lock_path.exists():
            return None

        manifest_path = lock_path.parent / "pixi.toml"
        if not manifest_path.exists():
            msg = f"Found {lock_path.name} but no pixi.toml next to it."
            raise RuntimeError(msg)

        for line in lock_path.read_text().splitlines():
            if line.lstrip().startswith("- pypi:"):
                msg = (
                    f"{lock_path} contains pypi dependencies. "
                    "misen owns PyPI packages through pyproject.toml / uv.lock; "
                    "remove them from the pixi manifest."
                )
                raise RuntimeError(msg)

        self.pixi_bin = shutil.which("pixi")
        if self.pixi_bin is None:
            msg = (
                "A pixi.lock was detected but the `pixi` CLI is not on PATH. "
                "Install it from https://pixi.sh to use conda dependencies with misen."
            )
            raise RuntimeError(msg)

        staged_manifest = snapshot_dir / "pixi.toml"
        staged_lock = snapshot_dir / "pixi.lock"
        shutil.copy(manifest_path, staged_manifest)
        shutil.copy(lock_path, staged_lock)

        try:
            subprocess.run(  # noqa: S603
                [
                    self.pixi_bin,
                    "--no-progress",
                    "--color",
                    "never",
                    "install",
                    "--frozen",
                    "--manifest-path",
                    str(staged_manifest),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or exc.stdout or "").strip()
            msg = f"pixi install failed for {lock_path}: {stderr}"
            raise RuntimeError(msg) from None

        return staged_manifest

    def _snapshot_env_files(self, snapshot_dir: Path) -> list[Path]:
        """Copy supported env files into snapshot directory.

        Args:
            snapshot_dir: Snapshot root directory.

        Returns:
            List of copied env-file paths.
        """
        env_files = []
        for src in _ENV_FILES:
            if src.exists():
                dst = snapshot_dir / src.name
                shutil.copy(src, dst)
                env_files.append(dst)
                # Restrict local override file permissions (likely to contain secrets).
                if src.name == ".env.local":
                    with contextlib.suppress(OSError):
                        dst.chmod(0o600)
        return env_files


@contextmanager
def apply_env_files_temporarily() -> Iterator[None]:
    """Temporarily load environment variables from dotenv files.

    Later files override earlier ones. Modified keys are restored after exiting
    the context.
    """
    initial_environ = os.environ.copy()
    for f in _ENV_FILES:
        if f.exists():
            load_dotenv(f, override=True)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(initial_environ)


_ENV_FILES = [Path.cwd() / name for name in (".env", ".env.local")]


def token_base32(nbytes: int) -> str:
    """Return URL/file-safe random base32 token.

    Args:
        nbytes: Number of random bytes before encoding.

    Returns:
        Base32 token without padding.
    """
    return base64.b32encode(secrets.token_bytes(nbytes)).decode("ascii").rstrip("=")


def _encode_cli_blob(payload: bytes) -> str:
    """Encode binary payload for safe CLI transport."""
    return base64.urlsafe_b64encode(payload).decode("ascii")
