"""Execution environment snapshots used by executors.

Snapshots capture enough environment state to run work units reproducibly in
subprocesses or on remote schedulers:

- isolated virtual environment (uv)
- optional conda prefix (installed and activated via the ``pixi`` CLI from
  ``pixi.lock`` + ``pixi.toml``)
- copied env files
- serialized callable payloads

Environments are the expensive part (a locked ML stack is gigabytes across
tens of thousands of files, and workspaces often live on NFS), so by default
they come from a *shared store* next to the snapshots directory: one
immutable, content-keyed entry per distinct lockfile state, built once and
reused by every subsequent snapshot. Each snapshot then only builds a small
overlay venv holding the project's local packages — the part that actually
changes between submissions. See :func:`_ensure_shared_entry` for the
crash-safe publication protocol.
"""

from __future__ import annotations

import base64
import contextlib
import logging
import os
import platform
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import cache
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import uv
from dotenv import load_dotenv

from misen.exceptions import LockUnavailableError
from misen.utils.fsync import fsync_dir
from misen.utils.hashing.base import hash_values
from misen.utils.locks import NFSLock
from misen.utils.runtime_events import runtime_event

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from misen.task_metadata import GpuRuntime
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = ["LocalSnapshot", "NullSnapshot", "Snapshot", "apply_env_files_temporarily"]

logger = logging.getLogger(__name__)

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
        gpu_runtime: GpuRuntime,
        *,
        cpu_indices: list[int] | None,
        gpu_indices: list[int] | None,
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
            gpu_runtime: Runtime environment for GPU resources.
            cpu_indices: CPU logical-core indices to bind in the worker via
                ``os.sched_setaffinity``. Pass ``None`` when the scheduler
                already pins CPUs (e.g. SLURM).
            gpu_indices: GPU device indices to mask in the worker via the
                runtime's visibility environment variables. Pass ``None`` when
                the scheduler already masks GPUs (e.g. SLURM cgroups).

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
        gpu_runtime: GpuRuntime,
        *,
        cpu_indices: list[int] | None,
        gpu_indices: list[int] | None,
    ) -> tuple[str, list[str], dict[str, str], Path]:
        """Prepare argv to execute the payload via ``uv run --no-project``.

        Args:
            work_unit: Work unit to execute.
            workspace: Workspace for payload/log paths.
            gpu_runtime: Runtime environment for GPU resources.
            cpu_indices: CPU logical-core indices for worker affinity, or
                ``None`` to leave inherited affinity untouched.
            gpu_indices: GPU device indices for worker visibility, or ``None``
                to leave inherited visibility untouched.

        Returns:
            Tuple ``(job_id, argv, env_overrides, log_path)``.
        """
        job_id = token_base32(6)

        if self.payload_dir is None:
            self.payload_dir = workspace.get_temp_dir() / "null_snapshot_payloads" / token_base32(6)
            self.payload_dir.mkdir(parents=True, exist_ok=True)

        payload_path = self.payload_dir / f"{token_base32(6)}.pkl"
        payload_path.write_bytes(work_unit.as_payload(workspace=workspace, job_id=job_id))

        log_path = workspace.get_job_log(job_id=job_id, work_unit=work_unit)
        argv = [
            *_detect_pixi_wrap(),
            *_uv_execute_argv(
                _active_env_files(),
                payload_path,
                gpu_runtime,
                cpu_indices=cpu_indices,
                gpu_indices=gpu_indices,
            ),
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

    With ``env_cache`` (the default), locked dependencies live in a shared
    content-keyed environment under ``<snapshots_dir>/.shared`` that is
    reused across snapshots and submissions; the snapshot itself only holds
    a small overlay venv with the project's local packages. Setting
    ``env_cache=False`` restores fully standalone per-snapshot environments.
    """

    __slots__ = (
        "conda_manifest_path",
        "env_files",
        "overlay_site_dir",
        "payload_dir",
        "pixi_bin",
        "python_env_dir",
        "shared_env_dir",
        "snapshot_dir",
    )

    def __init__(self, snapshots_dir: Path, *, env_cache: bool = True) -> None:
        """Create snapshot directory and materialize environment state.

        Args:
            snapshots_dir: Parent directory where snapshots are stored.
            env_cache: Whether locked dependencies may come from the shared
                env store under ``snapshots_dir / ".shared"`` (built once per
                lockfile state, reused across snapshots). When ``False``,
                every environment is built standalone inside the snapshot.
        """
        self.snapshot_dir = snapshots_dir / f"{token_base32(6)}"
        self.snapshot_dir.mkdir(parents=True)

        self.payload_dir = self.snapshot_dir / "payloads"
        self.payload_dir.mkdir(exist_ok=True)

        self.shared_env_dir: Path | None = None
        self.overlay_site_dir: Path | None = None
        self.python_env_dir = self._snapshot_python_env(env_cache=env_cache)
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
        gpu_runtime: GpuRuntime,
        *,
        cpu_indices: list[int] | None,
        gpu_indices: list[int] | None,
    ) -> tuple[str, list[str], dict[str, str], Path]:
        """Prepare command/env overrides to execute serialized payload.

        Args:
            work_unit: Work unit to execute.
            workspace: Workspace for payload/log paths.
            gpu_runtime: Runtime environment for GPU resources.
            cpu_indices: CPU logical-core indices for worker affinity, or
                ``None`` to leave inherited affinity untouched.
            gpu_indices: GPU device indices for worker visibility, or ``None``
                to leave inherited visibility untouched.

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

        argv += _uv_execute_argv(
            self.env_files,
            payload_path,
            gpu_runtime,
            cpu_indices=cpu_indices,
            gpu_indices=gpu_indices,
        )

        log_path = workspace.get_job_log(job_id=job_id, work_unit=work_unit)
        argv += [JOB_LOG_PATH_ARG, str(log_path)]

        env_overrides: dict[str, str] = {"VIRTUAL_ENV": str(self.python_env_dir)}
        if self.shared_env_dir is not None and self.overlay_site_dir is not None:
            # ``uv run`` only prepends the overlay's bin; the shared env's bin
            # (dependency console scripts like torchrun) must be added here.
            path_value = str(self.shared_env_dir / "bin")
            if os.environ.get("PATH"):
                path_value += os.pathsep + os.environ["PATH"]
            env_overrides["PATH"] = path_value
            # Safety net for children of shared-env scripts: their shebangs
            # point at the shared python, which never reads the overlay's
            # ``.pth`` — the inherited PYTHONPATH keeps local packages
            # importable there.
            pythonpath = str(self.overlay_site_dir)
            if os.environ.get("PYTHONPATH"):
                pythonpath += os.pathsep + os.environ["PYTHONPATH"]
            env_overrides["PYTHONPATH"] = pythonpath

        return job_id, argv, env_overrides, log_path

    def _snapshot_python_env(self, *, env_cache: bool) -> Path:
        """Materialize the snapshot's python environment.

        Default path: refresh the lockfile (matching the auto-lock behavior
        of a bare ``uv sync``), resolve the project interpreter, ensure the
        shared dependency env for the resulting content key exists, and build
        the per-snapshot overlay venv on top of it. Falls back to a
        standalone in-snapshot environment when ``env_cache`` is off or the
        interpreter cannot be resolved yet (e.g. a pinned python that uv has
        not installed — the standalone build auto-installs it, so the shared
        store takes over from the next snapshot on).

        Returns:
            The venv directory jobs should activate (``VIRTUAL_ENV``).

        Raises:
            RuntimeError: If any uv invocation fails.
        """
        if not env_cache:
            return self._snapshot_python_env_standalone(self.snapshot_dir / "python-env")

        _run_tool([_uv_bin(), "lock"], error_msg="Lockfile resolution (uv lock) failed")
        python = _uv_python_find()
        if python is None:
            logger.warning(
                "Could not resolve the project interpreter via `uv python find`; "
                "building a standalone snapshot environment instead of using the shared env store."
            )
            return self._snapshot_python_env_standalone(self.snapshot_dir / "python-env")

        store_root = self.snapshot_dir.parent / _SHARED_STORE_NAME
        cache_env = _uv_cache_env(store_root)

        def build(env_dir: Path) -> None:
            env = os.environ.copy() | cache_env | {"UV_PROJECT_ENVIRONMENT": str(env_dir)}
            _run_tool(
                [_uv_bin(), "sync", "--frozen", "--no-install-local", "--compile-bytecode"],
                env=env,
                error_msg="Shared virtual environment creation failed",
            )

        self.shared_env_dir = _ensure_shared_entry(
            store=store_root / "python-envs",
            key=_python_env_key(python),
            build=build,
            sanity_path="pyvenv.cfg",
            label="python env",
        )
        return self._snapshot_overlay_venv(self.shared_env_dir)

    def _snapshot_python_env_standalone(self, python_env_dir: Path) -> Path:
        """Install a frozen dependency snapshot into a standalone virtual env.

        The pre-shared-store build: everything (locked dependencies *and*
        the project's local packages, installed non-editably) goes into one
        venv inside the snapshot directory. Used when ``env_cache=False`` or
        the shared store is unavailable.

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
        _run_tool(
            [uv_bin, "sync", "--no-install-workspace"],
            env=env,
            error_msg="Virtual environment creation failed",
        )
        _run_tool(
            [uv_bin, "sync", "--no-editable", "--no-cache"],
            env=env,
            error_msg="Virtual environment creation failed",
        )
        return python_env_dir

    def _snapshot_overlay_venv(self, shared_env_dir: Path) -> Path:
        """Build the per-snapshot overlay venv chained to the shared env.

        The overlay is what jobs activate. It holds only the project's local
        packages (root, workspace members, path dependencies), each built
        fresh from the working tree and installed non-editably, plus a
        ``.pth`` entry extending ``sys.path`` into the shared env's
        site-packages. Local packages therefore shadow the shared env, any
        runtime installs land in the throwaway overlay rather than the
        shared env, and entry-point scripts of local packages get real
        launchers in ``<overlay>/bin``.

        Args:
            shared_env_dir: Completed shared env store entry to chain to.

        Returns:
            The overlay venv directory.

        Raises:
            RuntimeError: If any uv invocation fails.
        """
        venv_dir = self.snapshot_dir / "venv"
        _run_tool(
            # Resolves to the shared env's *base* interpreter (uv reads the
            # venv's pyvenv.cfg), so both envs run the identical python.
            [_uv_bin(), "venv", str(venv_dir), "--python", str(shared_env_dir / "bin" / "python")],
            error_msg="Overlay virtual environment creation failed",
        )
        result = _run_tool(
            [
                str(venv_dir / "bin" / "python"),
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            error_msg="Overlay site-packages resolution failed",
        )
        overlay_site = Path(result.stdout.strip())
        shared_site = shared_env_dir / overlay_site.relative_to(venv_dir)
        overlay_site.mkdir(parents=True, exist_ok=True)
        (overlay_site / "_misen_shared_env.pth").write_text(f"{shared_site}\n")

        local_packages = _local_package_paths(Path.cwd() / "uv.lock")
        if local_packages:
            _run_tool(
                [
                    _uv_bin(),
                    "pip",
                    "install",
                    "--python",
                    str(venv_dir / "bin" / "python"),
                    "--no-deps",
                    "--no-cache",
                    "--compile-bytecode",
                    *(str(p) for p in local_packages),
                ],
                error_msg="Local package installation failed",
            )
        self.overlay_site_dir = overlay_site
        return venv_dir

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


# --------------------------------------------------------------------------
# Shared env store
#
# One immutable directory per content key under ``<snapshots_dir>/.shared``
# (``token_base32`` snapshot names use only A-Z2-7, so ``.shared`` can never
# collide). Entries are published with a payload-before-pointer protocol:
# the ``<key>.complete`` marker file beside the entry is the commit point,
# written only after a successful build, so an entry without a marker is
# always crashed-builder residue.
# --------------------------------------------------------------------------

_SHARED_STORE_NAME = ".shared"
# Bump to invalidate every shared-store key (layout or build-flag changes).
_ENV_STORE_SCHEMA = 1
# Lock parameters for multi-minute builds. Waiters on other NFS clients read
# the lockfile mtime through their attribute cache (commonly up to 60s
# stale), so the lifetime/refresh headroom must exceed that staleness or a
# waiter could break a live builder's lease mid-build.
_BUILD_LOCK_LIFETIME_S = 120
_BUILD_LOCK_REFRESH_S = 30


def _run_tool(
    argv: list[str], *, env: dict[str, str] | None = None, error_msg: str
) -> subprocess.CompletedProcess[str]:
    """Run a CLI tool, wrapping failures in a ``RuntimeError`` with its output."""
    try:
        return subprocess.run(argv, check=True, capture_output=True, text=True, env=env)  # noqa: S603
    except subprocess.CalledProcessError as e:
        msg = f"{error_msg}: {(e.stderr or e.stdout or '').strip()}"
        raise RuntimeError(msg) from None


def _uv_python_find() -> str | None:
    """Return the resolved project interpreter path, or ``None`` if unresolved.

    ``--resolve-links`` yields the concrete interpreter installation (uv's
    symlink targets encode version, platform, and install root), which is
    exactly what the shared-env key must capture: two users pinning the same
    version but resolving different interpreter installs must not share.
    """
    try:
        result = subprocess.run(  # noqa: S603
            [_uv_bin(), "python", "find", "--resolve-links"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _store_key(parts: tuple[object, ...]) -> str:
    """Hash a tuple of key components to an unpadded-base32 store key."""
    digest = hash_values(parts)
    return base64.b32encode(digest.to_bytes(8, "big")).decode("ascii").rstrip("=")


def _python_env_key(python: str) -> str:
    """Content key of the shared python env for the project in CWD.

    Captures everything that determines the built env: the full dependency
    resolution (``uv.lock`` bytes), the interpreter identity (resolved path
    plus the ``.python-version`` pin, which can select differently once more
    interpreters are installed), and the build platform.
    """
    project_root = Path.cwd()
    pin = project_root / ".python-version"
    return _store_key(
        (
            _ENV_STORE_SCHEMA,
            (project_root / "uv.lock").read_bytes(),
            pin.read_bytes() if pin.exists() else None,
            python,
            sys.platform,
            platform.machine(),
        )
    )


def _existing_ancestor(path: Path) -> Path:
    """Return ``path`` or its nearest existing ancestor."""
    path = path.absolute()
    while not path.exists():
        if path.parent == path:
            break
        path = path.parent
    return path


def _same_filesystem(a: Path, b: Path) -> bool:
    """Whether two paths (or their nearest existing ancestors) share a device."""
    try:
        return _existing_ancestor(a).stat().st_dev == _existing_ancestor(b).stat().st_dev
    except OSError:
        return False


@cache
def _uv_cache_env(store_root: Path) -> dict[str, str]:
    """Return env additions implementing the uv cache-dir policy.

    uv hardlinks wheels from its cache into environments only when both sit
    on one filesystem; otherwise it silently falls back to copying every
    file — the dominant cost of env materialization on NFS workspaces. The
    policy: an explicit ``UV_CACHE_DIR`` is always respected; an effective
    cache (env default or uv config file) already on the store's filesystem
    is left alone (it is warm); only a cross-filesystem effective cache is
    replaced with one co-located in the store.
    """
    if "UV_CACHE_DIR" in os.environ:
        explicit = Path(os.environ["UV_CACHE_DIR"])
        if not _same_filesystem(explicit, store_root):
            logger.info(
                "UV_CACHE_DIR=%s is on a different filesystem than the snapshot env store %s; "
                "environment materialization will copy files instead of hardlinking.",
                explicit,
                store_root,
            )
        return {}

    try:
        result = subprocess.run(  # noqa: S603
            [_uv_bin(), "cache", "dir"], check=True, capture_output=True, text=True
        )
        effective = Path(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        effective = None
    if effective is not None and _same_filesystem(effective, store_root):
        return {}

    co_located = store_root / "uv-cache"
    logger.info(
        "uv cache (%s) is on a different filesystem than the snapshot env store; using a "
        "co-located cache at %s so environments hardlink instead of copy "
        "(export UV_CACHE_DIR to override).",
        effective if effective is not None else "unresolved",
        co_located,
    )
    return {"UV_CACHE_DIR": str(co_located)}


def _ensure_shared_entry(
    *,
    store: Path,
    key: str,
    build: Callable[[Path], None],
    sanity_path: str,
    label: str,
) -> Path:
    """Return the shared store entry for ``key``, building it if necessary.

    Entries are immutable once published and are built *in place* at their
    final path (venvs bake absolute paths into scripts, so build-then-rename
    is not an option). Same-key builders across hosts are serialized by an
    :class:`NFSLock`; the completion marker is written only after ``build``
    succeeds and only while the lease is still held, so readers can trust
    ``marker ⇒ complete entry`` and treat an unmarked entry as removable
    residue. Dependence on build ordering is safe because executors dispatch
    jobs only after snapshot creation returns — no job ever references an
    entry whose builder crashed before publishing.

    Args:
        store: Store directory holding entries, markers, and lock files.
        key: Content key naming the entry directory.
        build: Callback that materializes the entry at the given path.
        sanity_path: Entry-relative path that must exist for the entry to be
            considered usable (guards against a durable marker outliving an
            entry lost to an NFS-server crash).
        label: Human-readable entry kind for logs.

    Returns:
        The entry directory.

    Raises:
        RuntimeError: If the build fails or the build lock is lost mid-build.
    """
    entry_dir = store / key
    marker = store / f"{key}.complete"
    store.mkdir(parents=True, exist_ok=True)

    if marker.exists() and (entry_dir / sanity_path).exists():
        with contextlib.suppress(OSError):
            os.utime(marker)  # reuse breadcrumb for a future age-based prune
        logger.info("Reusing shared %s %s.", label, entry_dir)
        return entry_dir

    lock = NFSLock(
        lockfile=store / f"{key}.lock",
        lifetime=_BUILD_LOCK_LIFETIME_S,
        refresh_interval=_BUILD_LOCK_REFRESH_S,
    )
    try:
        lock.acquire(blocking=False)
    except LockUnavailableError:
        holder = lock.holder()
        held_by = f"{holder[0]} (pid {holder[1]})" if holder is not None else "another process"
        logger.info("Shared %s %s is being built by %s; waiting.", label, key, held_by)
        runtime_event(f"Waiting for a {label} build in progress on {held_by}", style="yellow")
        lock.acquire(blocking=True, timeout=None)
    try:
        _freshness_probe(store)
        if marker.exists():
            if (entry_dir / sanity_path).exists():
                with contextlib.suppress(OSError):
                    os.utime(marker)
                logger.info("Reusing shared %s %s (completed while waiting).", label, entry_dir)
                return entry_dir
            logger.warning(
                "Shared %s marker %s exists without a usable entry; rebuilding.", label, marker
            )
            marker.unlink()
            fsync_dir(store)
        if entry_dir.exists():
            logger.warning(
                "Removing incomplete shared %s %s left by an interrupted build.", label, entry_dir
            )
            _rmtree_with_retry(entry_dir)
        logger.info("Building shared %s %s.", label, entry_dir)
        build(entry_dir)
        if not lock.is_locked():
            # Lease stolen mid-build (extreme stall): a thief may already be
            # rebuilding this entry, so publishing our marker could bless a
            # half-built directory. Discard instead.
            msg = f"Lost the build lock for shared {label} {key}; discarding this build."
            raise RuntimeError(msg)
        _publish_marker(store, marker)
        return entry_dir
    finally:
        lock.release()


def _freshness_probe(directory: Path) -> None:
    """Create and unlink a temp file to force dentry revalidation in ``directory``.

    An NFS client's own directory mutation updates the cached change
    attribute, so the marker/entry re-checks made under the build lock don't
    act on stale (possibly negative) dentries — which could otherwise rmtree
    a complete entry as "residue".
    """
    with contextlib.suppress(OSError):
        fd, probe = tempfile.mkstemp(dir=directory, prefix=".misen.freshprobe.", suffix=".tmp")
        os.close(fd)
        Path(probe).unlink()


def _rmtree_with_retry(path: Path, attempts: int = 3) -> None:
    """``rmtree`` tolerating concurrent writes from an orphaned builder.

    A SIGKILLed builder can leave a uv/pixi child still writing into the
    entry; rmtree then races file creation and fails (e.g. ENOTEMPTY).
    Retry with backoff until the orphan exits or attempts run out.
    """
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
        else:
            return


def _publish_marker(store: Path, marker: Path) -> None:
    """Atomically publish a completion marker (payload-before-pointer commit).

    The entry's file *data* is already at the NFS server via close-to-open
    semantics when the builder's tool exits; per-file COMMITs over tens of
    thousands of files would defeat the store's purpose, so instead a single
    ``syncfs`` (where available) commits the mount's dirty pages, and the
    marker itself is fsync'd through the same mkstemp → fsync → rename →
    fsync-dir sequence used for hash-index writes.
    """
    if hasattr(os, "syncfs"):  # Linux
        with contextlib.suppress(OSError):
            fd = os.open(store, os.O_RDONLY)
            try:
                os.syncfs(fd)
            finally:
                os.close(fd)
    content = (
        f"host={socket.gethostname()} pid={os.getpid()} "
        f"time={time.time():.0f} schema={_ENV_STORE_SCHEMA}\n"
    )
    fd, tmp = tempfile.mkstemp(dir=store, prefix=f".{marker.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, marker)  # noqa: PTH105  -- atomic overwrite; Path has no equivalent
    finally:
        Path(tmp).unlink(missing_ok=True)
    fsync_dir(store)


def _local_package_paths(lock_path: Path) -> list[Path]:
    """Return local package paths recorded in ``uv.lock``, in lock order.

    These are exactly the packages ``uv sync --no-install-local`` skips when
    building the shared env: the root project, workspace members
    (``editable``), path dependencies (``directory``), and local
    wheel/sdist files (``path``). Their contents change with the working
    tree without changing the lockfile, so they belong to the per-snapshot
    overlay. ``virtual`` sources are never installed (only their
    dependencies are). Unrecognized source kinds fail loudly rather than
    risk baking stale code into a shared entry.

    Raises:
        RuntimeError: On an unsupported lockfile version or source kind.
    """
    lock = tomllib.loads(lock_path.read_text())
    lock_version = lock.get("version")
    if lock_version != 1:
        msg = f"Unsupported uv.lock version {lock_version!r} in {lock_path}; expected 1."
        raise RuntimeError(msg)
    paths: dict[Path, None] = {}
    for package in lock.get("package", []):
        source = package.get("source", {})
        local_kind = next((k for k in ("editable", "directory", "path") if k in source), None)
        if local_kind is not None:
            raw = Path(source[local_kind])
            resolved = raw if raw.is_absolute() else lock_path.parent / raw
            paths.setdefault(resolved.resolve(), None)
        elif not any(k in source for k in ("virtual", "registry", "git", "url")):
            msg = (
                f"Unrecognized source {source!r} for package {package.get('name')!r} in "
                f"{lock_path}; cannot classify it as local or remote."
            )
            raise RuntimeError(msg)
    return list(paths)


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
    gpu_runtime: GpuRuntime,
    *,
    cpu_indices: list[int] | None,
    gpu_indices: list[int] | None,
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
        "--gpu-runtime",
        gpu_runtime,
        *_indices_argv("cpu-indices", cpu_indices),
        *_indices_argv("gpu-indices", gpu_indices),
    ]


def _indices_argv(flag: str, indices: list[int] | None) -> list[str]:
    """Render ``--<flag> i j k`` (or ``--no-<flag>`` for None / [])."""
    if indices is None or len(indices) == 0:
        return []
    return [f"--{flag}", *(str(i) for i in indices)]


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
