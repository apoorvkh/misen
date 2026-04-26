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
from functools import cache
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

# CLI flag the worker entrypoint accepts (matches ``execute.execute``'s
# ``job_log_path`` parameter). Defined here rather than in
# ``misen.utils.execute`` so importing snapshot doesn't pre-load the worker
# module — the worker runs as ``python -m misen.utils.execute``, and the
# package-import chain (misen → executor → snapshot) would otherwise put
# ``misen.utils.execute`` in ``sys.modules`` before runpy executes it as
# ``__main__``, triggering a ``RuntimeWarning``.
JOB_LOG_PATH_ARG = "--job-log-path"


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
    ) -> tuple[str, list[str], dict[str, str], Path]:
        """Prepare command and environment for one work unit.

        ``argv`` includes the worker's ``--job-log-path`` argument so the
        worker can wrap its lifecycle in
        :meth:`Workspace.streaming_job_log` against the same file the
        executor parent will use for its own output redirection (the
        returned ``log_path``).

        Args:
            work_unit: Work unit to execute.
            workspace: Workspace for payload/log paths.
            assigned_resources_getter: Callable returning runtime resources for
                sentinel injection and worker CPU/GPU binding.
            gpu_runtime: Runtime environment for GPU resources.

        Returns:
            Tuple ``(job_id, argv, env_overrides, log_path)``.
        """


class NullSnapshot(Snapshot):
    """Snapshot that dispatches via ``uv run --no-project`` in the current env.

    Used by :class:`~misen.executors.in_process.InProcessExecutor` (in-process)
    and by ``LocalExecutor(snapshot=False)`` (subprocess dispatch). Skips uv
    venv materialization and env-file copying, so jobs start instantly — but
    they run against whatever interpreter and environment the parent process
    has, and are therefore sensitive to code or dependency edits made while
    the job runs. ``.env`` / ``.env.local`` are read live from CWD by
    ``uv run --env-file`` (no staged copy).

    When a ``pixi.toml`` sits in CWD and the ``pixi`` CLI is on PATH,
    subprocess dispatch wraps argv in ``pixi run --frozen -x -- ...``
    against the in-tree manifest, so conda activation still runs against
    the locked env without any install work on the dispatch path.
    """

    __slots__ = ("payload_dir",)

    def __init__(self) -> None:
        """Initialize with a lazily-allocated payload directory.

        Eagerly invokes :func:`_detect_pixi_wrap` so a misconfigured
        ``pixi.lock`` fails at snapshot creation rather than dispatch.
        """
        self.payload_dir: Path | None = None
        _detect_pixi_wrap()

    def cleanup(self) -> None:
        """Remove payload directory if one was created during dispatch."""
        if self.payload_dir is not None:
            shutil.rmtree(self.payload_dir, ignore_errors=True)
            self.payload_dir = None

    def prepare_job(
        self,
        work_unit: WorkUnit,
        workspace: Workspace,
        assigned_resources_getter: Callable[[], AssignedResources | AssignedResourcesPerNode | None],
        gpu_runtime: GpuRuntime,
    ) -> tuple[str, list[str], dict[str, str], Path]:
        """Prepare argv to execute the payload via ``uv run --no-project``.

        Args:
            work_unit: Work unit to execute.
            workspace: Workspace for payload/log paths.
            assigned_resources_getter: Callable returning runtime resources for
                task sentinel injection and worker binding.
            gpu_runtime: Runtime environment for GPU resources.

        Returns:
            Tuple ``(job_id, argv, env_overrides, log_path)``.
        """
        job_id = token_base32(6)

        if self.payload_dir is None:
            self.payload_dir = workspace.get_temp_dir() / "null_snapshot_payloads" / token_base32(6)
            self.payload_dir.mkdir(parents=True, exist_ok=True)

        payload_path = self.payload_dir / f"{token_base32(6)}.pkl"
        payload_path.write_bytes(work_unit.as_payload(workspace=workspace, job_id=job_id))
        encoded_getter = _encode_cli_blob(cloudpickle.dumps(assigned_resources_getter))

        log_path = workspace.get_job_log(job_id=job_id, work_unit=work_unit)
        argv = [
            *_detect_pixi_wrap(),
            *_uv_execute_argv(_active_env_files(), payload_path, encoded_getter, gpu_runtime),
            JOB_LOG_PATH_ARG,
            str(log_path),
        ]

        return job_id, argv, {}, log_path


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
    ) -> tuple[str, list[str], dict[str, str], Path]:
        """Prepare command/env overrides to execute serialized payload.

        Args:
            work_unit: Work unit to execute.
            workspace: Workspace for payload/log paths.
            assigned_resources_getter: Callable returning runtime resources for
                task sentinel injection and worker binding.
            gpu_runtime: Runtime environment for GPU resources.

        Returns:
            Tuple ``(job_id, argv, env_overrides, log_path)``.
        """
        job_id = token_base32(6)

        argv: list[str] = []

        # When a conda env is present, wrap argv so pixi activates it at job spawn.
        if self.conda_manifest_path is not None and self.pixi_bin is not None:
            argv += _pixi_run_prefix(self.pixi_bin, self.conda_manifest_path)

        payload_path = self.payload_dir / f"{token_base32(6)}.pkl"
        payload_path.write_bytes(work_unit.as_payload(workspace=workspace, job_id=job_id))
        encoded_getter = _encode_cli_blob(cloudpickle.dumps(assigned_resources_getter))

        argv += _uv_execute_argv(self.env_files, payload_path, encoded_getter, gpu_runtime)

        log_path = workspace.get_job_log(job_id=job_id, work_unit=work_unit)
        argv += [JOB_LOG_PATH_ARG, str(log_path)]

        env_overrides: dict[str, str] = {"VIRTUAL_ENV": str(self.python_env_dir)}

        return job_id, argv, env_overrides, log_path

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
        uv_bin = _uv_bin()
        try:
            subprocess.run(  # noqa: S603
                [uv_bin, "sync", "--no-install-workspace"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            subprocess.run(  # noqa: S603
                [uv_bin, "sync", "--no-editable", "--no-cache"],
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

        _check_pixi_lock_for_pypi(lock_path)

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


@cache
def _active_env_files() -> tuple[Path, ...]:
    """Return ``.env`` / ``.env.local`` paths that exist in CWD (cached)."""
    return tuple(p for p in _ENV_FILES if p.exists())


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


def _check_pixi_lock_for_pypi(lock_path: Path) -> None:
    """Raise if ``pixi.lock`` contains PyPI entries.

    misen owns PyPI packages through ``pyproject.toml`` / ``uv.lock``, so a
    pixi lock that lists PyPI dependencies would double-install or conflict.

    Raises:
        RuntimeError: If any ``- pypi:`` entry is present in ``lock_path``.
    """
    for line in lock_path.read_text().splitlines():
        if line.lstrip().startswith("- pypi:"):
            msg = (
                f"{lock_path} contains pypi dependencies. "
                "misen owns PyPI packages through pyproject.toml / uv.lock; "
                "remove them from the pixi manifest."
            )
            raise RuntimeError(msg)


@cache
def _uv_bin() -> str:
    """Return the uv CLI path (cached — constant across the process)."""
    return uv.find_uv_bin()


def _uv_execute_argv(
    env_files: list[Path] | tuple[Path, ...],
    payload_path: Path,
    encoded_getter: str,
    gpu_runtime: GpuRuntime,
) -> list[str]:
    """Build the ``uv run --no-project -m misen.utils.execute ...`` argv.

    Shared by both snapshot classes; the only per-class difference is the
    ``env_files`` source (live CWD paths for :class:`NullSnapshot`, staged
    copies for :class:`LocalSnapshot`).
    """
    return [
        _uv_bin(),
        "run",
        "--no-project",
        *chain.from_iterable(("--env-file", str(path)) for path in env_files),
        "-m",
        "misen.utils.execute",
        "--payload",
        str(payload_path),
        "--assigned-resources-getter",
        encoded_getter,
        "--gpu-runtime",
        gpu_runtime,
    ]


def _pixi_run_prefix(pixi_bin: str, manifest_path: Path) -> list[str]:
    """Return ``pixi run --frozen -x -- …`` argv prefix for activation wrapping.

    ``-x`` forces executable mode (no pixi-task lookup); ``--`` stops pixi
    from parsing the wrapped command's flags.
    """
    return [
        pixi_bin,
        "run",
        "--no-progress",
        "--color",
        "never",
        "--frozen",
        "--manifest-path",
        str(manifest_path),
        "-x",
        "--",
    ]


@cache
def _detect_pixi_wrap() -> list[str]:
    """Return a ``pixi run`` argv prefix for in-tree activation, or ``[]``.

    Cached on the process — CWD and ``pixi.toml`` are assumed stable for
    misen's lifetime. Used by :class:`NullSnapshot` so subprocess dispatch
    still gets conda activation when the caller's project declares a pixi
    env. Unlike :class:`LocalSnapshot`, which copies ``pixi.toml`` /
    ``pixi.lock`` into the snapshot dir and pre-installs the env, this wrap
    runs pixi ``--frozen`` against the in-tree manifest — no install or
    resolution work happens at dispatch.

    Returns:
        argv prefix ending in ``--``, or ``[]`` when pixi isn't applicable.

    Raises:
        RuntimeError: If ``pixi.lock`` contains PyPI dependencies.
    """
    manifest_path = Path.cwd() / "pixi.toml"
    if not manifest_path.exists():
        return []
    pixi_bin = shutil.which("pixi")
    if pixi_bin is None:
        return []

    lock_path = Path.cwd() / "pixi.lock"
    if lock_path.exists():
        _check_pixi_lock_for_pypi(lock_path)

    return _pixi_run_prefix(pixi_bin, manifest_path)
