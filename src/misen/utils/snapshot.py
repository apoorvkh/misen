"""Execution environment snapshots used by executors.

Snapshots capture enough environment state to run work units reproducibly in
subprocesses or on remote schedulers:

- isolated virtual environment (uv)
- optional conda prefix (installed and activated via the ``pixi`` CLI from
  ``pixi.lock`` + ``pixi.toml``)
- copied env files
- serialized callable payloads

Environments are the expensive part (a locked ML stack is gigabytes across
tens of thousands of files, often on NFS), so both the uv dependency env and
the optional conda env come from a content-keyed **env store** — built once
per lockfile state and reused across snapshots — while each snapshot only
builds a small overlay venv holding the project's local packages. Protocol
rationale: ``docs/design_shared_env_store.md``.
"""

from __future__ import annotations

import base64
import contextlib
import json
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
    CWD, a parallel conda env is installed via ``pixi install --frozen``.
    Jobs are then wrapped in ``pixi run --frozen -x -- <uv run ...>`` so
    activation (``CONDA_PREFIX``, ``PATH``, ``LD_LIBRARY_PATH``, plus
    anything ``activate.d`` scripts inject like ``CUDA_HOME``) happens at
    job spawn. The conda prefix supplies native / system libraries while
    Python and every PyPI package stay in the python env.

    Locked dependencies come from content-keyed env-store entries and the
    snapshot itself only builds a small overlay venv with the project's
    local packages. With ``env_cache`` (the default) the store lives at
    ``<snapshots_dir>/.shared`` and entries are reused across snapshots
    and submissions; with ``env_cache=False`` it is private to this
    snapshot (nothing shared, removed by :meth:`cleanup`).
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
            env_cache: Whether env-store entries are shared across snapshots
                (built once per lockfile state) or private to this snapshot.
        """
        self.snapshot_dir = snapshots_dir / f"{token_base32(6)}"
        self.snapshot_dir.mkdir(parents=True)

        self.payload_dir = self.snapshot_dir / "payloads"
        self.payload_dir.mkdir(exist_ok=True)

        self.pixi_bin: str | None = None
        self.python_env_dir = self._snapshot_python_env(env_cache=env_cache)
        self.conda_manifest_path = self._snapshot_conda(env_cache=env_cache)
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

        # ``uv run`` prepends only the overlay venv's bin; the deps env's bin
        # carries dependency console scripts (e.g. torchrun), and PYTHONPATH
        # keeps local packages importable for children of deps-env script
        # shebangs (which never read the overlay's ``.pth``).
        env_overrides = {
            "VIRTUAL_ENV": str(self.python_env_dir),
            "PATH": _prepend_path_var(str(self.shared_env_dir / "bin"), os.environ.get("PATH")),
            "PYTHONPATH": _prepend_path_var(str(self.overlay_site_dir), os.environ.get("PYTHONPATH")),
        }

        return job_id, argv, env_overrides, log_path

    def _env_store_root(self, *, env_cache: bool) -> Path:
        """Env-store root: shared beside all snapshots, or private inside this one."""
        return self.snapshot_dir.parent / _SHARED_STORE_NAME if env_cache else self.snapshot_dir / "envs"

    def _snapshot_python_env(self, *, env_cache: bool) -> Path:
        """Materialize the python env: deps store entry + overlay venv.

        ``uv lock`` first preserves the auto-lock behavior of the bare
        ``uv sync`` this replaced; the locked dependencies then come from
        an env-store entry and only the overlay venv is built per snapshot.

        Returns:
            The venv directory jobs should activate (``VIRTUAL_ENV``).

        Raises:
            RuntimeError: If any uv invocation fails.
        """
        _run_tool([_uv_bin(), "lock"], error_msg="Lockfile resolution (uv lock) failed")
        store_root = self._env_store_root(env_cache=env_cache)
        cache_env = _uv_cache_env(store_root) if env_cache else {}

        def build(env_dir: Path) -> None:
            env = os.environ.copy() | cache_env | {"UV_PROJECT_ENVIRONMENT": str(env_dir)}
            _run_tool(
                [_uv_bin(), "sync", "--frozen", "--no-install-local", "--compile-bytecode"],
                env=env,
                error_msg="Virtual environment creation failed",
            )

        self.shared_env_dir = _ensure_store_entry(
            store=store_root / "python-envs",
            key=_python_env_key(),
            build=build,
            # ``exists()`` follows symlinks, so an entry whose interpreter
            # was uninstalled reads as unusable and rebuilds.
            sanity_path="bin/python",
            label="python env",
        )
        return self._snapshot_overlay_venv(self.shared_env_dir)

    def _snapshot_overlay_venv(self, deps_env_dir: Path) -> Path:
        """Build the per-snapshot overlay venv chained to the deps env.

        The overlay is what jobs activate: the project's local packages,
        built fresh from the working tree and installed non-editably, plus
        a ``.pth`` extending ``sys.path`` into the deps env. Local packages
        shadow the deps env, runtime installs land in the throwaway overlay
        rather than the deps env, and local entry-point scripts get real
        launchers in ``<overlay>/bin``.

        Raises:
            RuntimeError: If any uv invocation fails.
        """
        venv_dir = self.snapshot_dir / "venv"
        _run_tool(
            # ``--python <deps env python>`` resolves to its *base* interpreter.
            [_uv_bin(), "venv", str(venv_dir), "--python", str(deps_env_dir / "bin" / "python")],
            error_msg="Overlay virtual environment creation failed",
        )
        overlay_site = next((venv_dir / "lib").glob("python*")) / "site-packages"
        deps_site = deps_env_dir / overlay_site.relative_to(venv_dir)
        (overlay_site / "_misen_shared_env.pth").write_text(f"{deps_site}\n")

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

    def _snapshot_conda(self, *, env_cache: bool) -> Path | None:
        """Install an optional conda env from ``pixi.lock`` via the pixi CLI.

        The staged ``pixi.toml`` + ``pixi.lock`` and the installed
        ``.pixi/envs/default`` prefix form one env-store entry keyed by the
        two manifests' bytes. Activation is deferred to job spawn:
        :meth:`prepare_job` wraps argv in ``pixi run --frozen -x -- ...``,
        and the python env still owns the interpreter at runtime because
        ``uv run`` prepends its bin ahead of the conda prefix on ``PATH``.

        Returns:
            Path to the staged ``pixi.toml`` (pixi's manifest-path flag
            consumes this), or ``None`` when no ``pixi.lock`` is in CWD.

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

        pixi_bin = self.pixi_bin = shutil.which("pixi")
        if pixi_bin is None:
            msg = (
                "A pixi.lock was detected but the `pixi` CLI is not on PATH. "
                "Install it from https://pixi.sh to use conda dependencies with misen."
            )
            raise RuntimeError(msg)

        store_root = self._env_store_root(env_cache=env_cache)
        cache_env = _pixi_cache_env(store_root) if env_cache else {}
        key = _store_key(
            (_ENV_STORE_SCHEMA, manifest_path.read_bytes(), lock_path.read_bytes(), sys.platform, platform.machine())
        )

        def build(entry_dir: Path) -> None:
            entry_dir.mkdir(parents=True)
            shutil.copy(manifest_path, entry_dir / "pixi.toml")
            shutil.copy(lock_path, entry_dir / "pixi.lock")
            _run_tool(
                [
                    pixi_bin,
                    "--no-progress",
                    "--color",
                    "never",
                    "install",
                    "--frozen",
                    "--manifest-path",
                    str(entry_dir / "pixi.toml"),
                ],
                env=os.environ.copy() | cache_env,
                error_msg=f"pixi install failed for {lock_path}",
            )

        entry_dir = _ensure_store_entry(
            store=store_root / "conda-envs",
            key=key,
            build=build,
            sanity_path=".pixi/envs/default",
            label="conda env",
        )
        return entry_dir / "pixi.toml"

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
# Env store: one immutable directory per content key, published with a
# payload-before-pointer protocol (the fsync'd ``<key>.complete`` marker is
# the commit point). See docs/design_shared_env_store.md.
# --------------------------------------------------------------------------

_SHARED_STORE_NAME = ".shared"  # snapshot tokens are A-Z2-7, so this never collides
_ENV_STORE_SCHEMA = 1  # bump to invalidate every store key
# Multi-minute builds: the lifetime-minus-refresh headroom must exceed NFS
# attribute-cache staleness (~60s), or a waiter reading a stale lockfile
# mtime could break a live builder's lease.
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


def _prepend_path_var(head: str, existing: str | None) -> str:
    """Prepend ``head`` to a ``os.pathsep``-separated env value."""
    return head if not existing else f"{head}{os.pathsep}{existing}"


def _store_key(parts: tuple[object, ...]) -> str:
    """Hash a tuple of key components to an unpadded-base32 store key."""
    digest = hash_values(parts)
    return base64.b32encode(digest.to_bytes(8, "big")).decode("ascii").rstrip("=")


def _python_env_key() -> str:
    """Content key of the deps env for the project in CWD.

    Captures the resolution (``uv.lock`` bytes), the interpreter-selection
    inputs (``.python-version`` pin and ``UV_PYTHON``), and the platform.
    An interpreter upgrade satisfying the same pin keeps the key — the
    entry stays self-consistent, and if its interpreter is ever uninstalled
    the ``bin/python`` sanity check fails and the entry rebuilds.
    """
    pin = Path.cwd() / ".python-version"
    return _store_key(
        (
            _ENV_STORE_SCHEMA,
            (Path.cwd() / "uv.lock").read_bytes(),
            pin.read_bytes() if pin.exists() else None,
            os.environ.get("UV_PYTHON"),
            sys.platform,
            platform.machine(),
        )
    )


def _same_filesystem(a: Path, b: Path) -> bool:
    """Whether two paths (or their nearest existing ancestors) share a device."""

    def dev(path: Path) -> int:
        path = path.absolute()
        while not path.exists() and path.parent != path:
            path = path.parent
        return path.stat().st_dev

    try:
        return dev(a) == dev(b)
    except OSError:
        return False


def _cache_dir_env(
    label: str,
    explicit_vars: tuple[str, ...],
    resolve: Callable[[], Path | None],
    subdir: str,
    store_root: Path,
) -> dict[str, str]:
    """Cache-dir policy for env builds.

    Linking from cache into an env requires both on one filesystem —
    otherwise uv/pixi silently fall back to copying every file. Explicit
    cache env vars always win; a same-filesystem effective cache is left
    alone (it is warm); only a cross-filesystem effective cache is replaced
    with one co-located in the store.
    """
    for var in explicit_vars:
        if var in os.environ:
            if not _same_filesystem(Path(os.environ[var]), store_root):
                logger.info(
                    "%s=%s is on a different filesystem than the env store %s; "
                    "%s env materialization will copy instead of link.",
                    var,
                    os.environ[var],
                    store_root,
                    label,
                )
            return {}
    effective = resolve()
    if effective is not None and _same_filesystem(effective, store_root):
        return {}
    co_located = store_root / subdir
    logger.info(
        "%s cache (%s) is on a different filesystem than the env store; using the "
        "co-located cache %s so envs link instead of copy (set %s to override).",
        label,
        effective if effective is not None else "unresolved",
        co_located,
        explicit_vars[0],
    )
    return {explicit_vars[0]: str(co_located)}


@cache
def _uv_cache_env(store_root: Path) -> dict[str, str]:
    """Return the uv cache-dir policy for ``store_root`` (cached per store)."""

    def resolve() -> Path | None:
        try:
            out = subprocess.run(  # noqa: S603
                [_uv_bin(), "cache", "dir"], check=True, capture_output=True, text=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return Path(out.stdout.strip())

    return _cache_dir_env("uv", ("UV_CACHE_DIR",), resolve, "uv-cache", store_root)


@cache
def _pixi_cache_env(store_root: Path) -> dict[str, str]:
    """Return the pixi cache-dir policy for ``store_root`` (cached per store)."""

    def resolve() -> Path | None:
        pixi_bin = shutil.which("pixi")
        if pixi_bin is None:
            return None
        try:
            out = subprocess.run(  # noqa: S603
                [pixi_bin, "info", "--json"], check=True, capture_output=True, text=True
            )
            cache_dir = json.loads(out.stdout).get("cache_dir")
        except (OSError, subprocess.CalledProcessError, ValueError):
            return None
        return Path(cache_dir) if cache_dir else None

    return _cache_dir_env("pixi", ("PIXI_CACHE_DIR", "RATTLER_CACHE_DIR"), resolve, "pixi-cache", store_root)


def _ensure_store_entry(
    *,
    store: Path,
    key: str,
    build: Callable[[Path], None],
    sanity_path: str,
    label: str,
) -> Path:
    """Return the store entry for ``key``, building it in place if necessary.

    Entries are immutable once published and built at their final path
    (venvs bake absolute paths into scripts, so build-then-rename is out).
    The fsync'd marker beside the entry is the commit point: written only
    after ``build`` succeeds and only while the lock lease is still held,
    so an entry without a marker is crashed-builder residue — safe to
    remove, because executors dispatch jobs only after snapshot creation
    returns. ``sanity_path`` (entry-relative) must exist for a marked entry
    to count as usable, healing a durable marker that outlived its entry.

    Raises:
        RuntimeError: If the build fails or the build lock is lost mid-build.
    """
    entry_dir = store / key
    marker = store / f"{key}.complete"
    store.mkdir(parents=True, exist_ok=True)

    def reuse_if_ready() -> Path | None:
        if not (marker.exists() and (entry_dir / sanity_path).exists()):
            return None
        with contextlib.suppress(OSError):
            os.utime(marker)  # reuse breadcrumb for a future age-based prune
        logger.info("Reusing %s %s.", label, entry_dir)
        return entry_dir

    if (entry := reuse_if_ready()) is not None:
        return entry

    lock = NFSLock(store / f"{key}.lock", lifetime=_BUILD_LOCK_LIFETIME_S, refresh_interval=_BUILD_LOCK_REFRESH_S)
    try:
        lock.acquire(blocking=False)
    except LockUnavailableError:
        holder = lock.holder()
        held_by = f"{holder[0]} (pid {holder[1]})" if holder else "another process"
        logger.info("%s %s is being built by %s; waiting.", label, key, held_by)
        runtime_event(f"Waiting for a {label} build in progress on {held_by}", style="yellow")
        lock.acquire(blocking=True, timeout=None)
    try:
        # Acquiring the lock wrote claim files into ``store``, refreshing this
        # client's (possibly negative) dentry cache for the re-checks below.
        if (entry := reuse_if_ready()) is not None:
            return entry
        if marker.exists():
            logger.warning("%s marker %s exists without a usable entry; rebuilding.", label, marker)
            marker.unlink()
            fsync_dir(store)
        if entry_dir.exists():
            logger.warning("Removing incomplete %s %s left by an interrupted build.", label, entry_dir)
            _rmtree_with_retry(entry_dir)
        logger.info("Building %s %s.", label, entry_dir)
        build(entry_dir)
        if not lock.is_locked():
            # Lease stolen mid-build (extreme stall): a thief may already be
            # rebuilding this entry, so our marker could bless a half-built one.
            msg = f"Lost the build lock for {label} {key}; discarding this build."
            raise RuntimeError(msg)
        _publish_marker(store, marker)
        return entry_dir
    finally:
        lock.release()


def _rmtree_with_retry(path: Path, attempts: int = 3) -> None:
    """``rmtree`` with backoff, tolerating writes from an orphaned builder's child."""
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

    Entry file data already reached the NFS server via close-to-open when
    the build tool exited; one ``syncfs`` commits the mount's dirty pages
    (per-file COMMITs over tens of thousands of files would defeat the
    store's purpose), then the marker is published with the same durable
    mkstemp → fsync → rename → fsync-dir sequence used for hash-index writes.
    """
    if hasattr(os, "syncfs"):  # Linux
        with contextlib.suppress(OSError):
            fd = os.open(store, os.O_RDONLY)
            try:
                os.syncfs(fd)
            finally:
                os.close(fd)
    fd, tmp = tempfile.mkstemp(dir=store, prefix=f".{marker.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(f"host={socket.gethostname()} pid={os.getpid()} time={time.time():.0f}\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, marker)  # noqa: PTH105  -- atomic overwrite; Path has no equivalent
    finally:
        Path(tmp).unlink(missing_ok=True)
    fsync_dir(store)


def _local_package_paths(lock_path: Path) -> list[Path]:
    """Return local package paths recorded in ``uv.lock``, in lock order.

    Exactly the packages ``uv sync --no-install-local`` skips: the root
    project and workspace members (``editable``), path dependencies
    (``directory``), and local wheel/sdist files (``path``). They change
    with the working tree without changing the lockfile, so they belong to
    the per-snapshot overlay. ``virtual`` sources are never installed.
    Unrecognized source kinds fail loudly rather than risk baking stale
    code into a store entry.

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
            paths.setdefault((raw if raw.is_absolute() else lock_path.parent / raw).resolve(), None)
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
