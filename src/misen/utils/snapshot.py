"""Execution snapshots: content-addressed project state + env materialization.

A :class:`ProjectSnapshot` is pure data — the project's code (local packages
built to wheels/sdists) and uv/pixi dependency metadata — staged,
content-hashed, and published into the workspace's snapshot store.
Environments are *materialized from* snapshots by :func:`_materialize_envs`
into a content-keyed **env store** (built once per content state per store,
with crash-safe locking), either on the submitting host at snapshot time
(prewarm) or on each execution host at job startup (through the submitted
Bash bootstrap). Design rationale:
``docs/design_shared_env_store.md`` and ``docs/design_unified_snapshot.md``.

Submission-scoped secrets (``.env`` files) and per-job payloads are never
part of the content-addressed snapshot; they travel through the
workspace's job-file store.
"""

from __future__ import annotations

import base64
import contextlib
import getpass
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
from contextlib import contextmanager
from functools import cache
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit

import uv
from dotenv import load_dotenv

from misen.exceptions import LockUnavailableError
from misen.utils.bootstrap_transport import worker_bootstrap_script
from misen.utils.fsync import fsync_dir
from misen.utils.hashing import Hash
from misen.utils.hashing.base import hash_values
from misen.utils.locks import NFSLock
from misen.utils.runtime_events import runtime_event

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from misen.task_metadata import GpuRuntime
    from misen.utils.work_unit import WorkUnit
    from misen.workspace import Workspace

__all__ = ["ProjectSnapshot", "apply_env_files_temporarily", "prepare_live_job"]

logger = logging.getLogger(__name__)

# CLI flag the worker entrypoint accepts (matches ``execute.execute``'s
# ``job_log_path`` parameter). Defined here rather than in
# ``misen.utils.execute`` so importing snapshot doesn't pre-load the worker
# module — the worker runs as ``python -m misen.utils.execute``, and the
# package-import chain (misen → executor → snapshot) would otherwise put
# ``misen.utils.execute`` in ``sys.modules`` before runpy executes it as
# ``__main__``, triggering a ``RuntimeWarning``.
JOB_LOG_PATH_ARG = "--job-log-path"


def prepare_live_job(
    work_unit: WorkUnit,
    workspace: Workspace,
    gpu_runtime: GpuRuntime,
    *,
    cpu_indices: list[int] | None,
    gpu_indices: list[int] | None,
) -> tuple[str, list[str], dict[str, str], Path]:
    """Prepare a work unit to run via ``uv run --no-project`` in the live env.

    The ``snapshot = false`` dispatch mode: no snapshot is taken, so jobs
    start instantly but run against whatever interpreter and environment
    the parent process has — sensitive to code or dependency edits made
    while the job runs. ``.env`` / ``.env.local`` are read live from CWD,
    and when a ``pixi.toml`` sits in CWD (with the ``pixi`` CLI on PATH)
    argv is wrapped in ``pixi run --frozen -- …`` against the in-tree
    manifest, so conda activation still applies with no install work.

    As in :meth:`ProjectSnapshot.prepare_job`, ``argv`` carries the
    worker's ``--job-log-path`` so it can wrap its lifecycle in
    :meth:`Workspace.streaming_job_log` against the same file the executor
    uses for output redirection (the returned ``log_path``). Payloads land
    in the workspace's submission-scoped job-file store and are retained
    until workspace pruning or manual removal.

    Args:
        work_unit: Work unit to execute.
        workspace: Workspace for payload/log paths.
        gpu_runtime: Runtime environment for GPU resources.
        cpu_indices: CPU logical-core indices for worker affinity, or
            ``None`` when the scheduler already pins CPUs (e.g. SLURM).
        gpu_indices: GPU device indices for worker visibility, or ``None``
            when the scheduler already masks GPUs (e.g. SLURM cgroups).

    Returns:
        Tuple ``(job_id, argv, env_overrides, log_path)``.
    """
    job_id = token_base32(6)

    payload_path = workspace.get_temp_dir() / "live_payloads" / f"{job_id}.pkl"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
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


class ProjectSnapshot:
    """Content-addressed snapshot of the caller's project, published to the workspace.

    A snapshot is pure data — code and dependency metadata, never a built
    environment:

    - ``packages/``: each local package built from the working tree now,
      so queued jobs stay pinned to the submitted code. Pure-python wheels
      are staged as wheels; native packages are staged as sdists and
      compile on the execution host inside the pixi activation (where the
      locked toolchain and the correct platform exist);
    - ``pyproject.toml`` / ``uv.lock`` / ``.python-version``: the frozen uv
      project used to sync remote dependencies without installing local
      packages, including the project's source and index configuration;
    - ``pixi.toml`` + ``pixi.lock`` when the project has a conda env.

    The staged tree is hashed and published once per content key into the
    workspace snapshot store (:meth:`Workspace.publish_snapshot`), which is
    the data plane for code distribution. Environments are materialized
    from it by :func:`_materialize_envs` into a content-keyed env store —
    either on the submitting host at snapshot time (``prewarm=True``, so
    jobs dispatch with direct activation and workers need nothing but the
    store) or on each execution host at job startup via
    one submitted Bash bootstrap (``prewarm=False``, so workers build into
    their own local disk).

    Submission-scoped state that must *not* be content-addressed — copies
    of ``.env`` / ``.env.local`` (secrets) and per-job payloads — goes
    through the workspace's job-file store instead, grouped under this
    snapshot's ``submission_id``. Nothing is deleted per submission:
    payloads must outlive scheduler requeues, so job files (like the
    snapshot and env-store entries) are retained for a future age-based
    prune.
    """

    __slots__ = (
        "env_file_refs",
        "env_store_dir",
        "misen_requirement",
        "pixi_bin",
        "prewarmed",
        "project_dir",
        "snapshot_key",
        "submission_id",
        "transport",
        "workspace",
    )

    def __init__(
        self,
        workspace: Workspace,
        *,
        env_store_dir: str | None = None,
        prewarm: bool = False,
    ) -> None:
        """Stage the project, publish it, and optionally prewarm environments.

        Args:
            workspace: Workspace that stores the snapshot and job files.
            env_store_dir: Env-store root where environments materialize
                (must be node-local disk for worker-side builds, or a
                shared filesystem when prewarmed envs must be visible to
                remote workers). ``None`` uses the per-user default
                (``/tmp/misen-env-store-<user>``) on whichever host builds.
            prewarm: Materialize the environments now, on this host, into
                ``env_store_dir``. Jobs then dispatch with direct
                activation (no worker-side bootstrap), restoring
                fail-fast environment errors at submission time.

        Raises:
            RuntimeError: If staging fails (uv/pixi invocations, invalid
                pixi manifests) or a prewarm build fails.
        """
        self.transport = workspace.bootstrap_transport()
        if self.transport is not None and (not isinstance(self.transport, str) or not self.transport.strip()):
            msg = "bootstrap_transport() must return non-empty Bash source or None for path transport."
            raise ValueError(msg)
        if prewarm and self.transport is not None:
            msg = (
                "prewarm_envs requires a workspace with worker-visible paths for snapshots and job files "
                f"({type(workspace).__name__} declares a transport); set prewarm_envs=False."
            )
            raise ValueError(msg)

        self.workspace = workspace
        self.env_store_dir = env_store_dir
        self.submission_id = token_base32(6)
        self.pixi_bin: str | None = None

        staged = workspace.get_temp_dir() / "snapshot_staging" / token_base32(6)
        try:
            self._stage(staged)
            self.snapshot_key = _snapshot_key(staged)
            workspace.publish_snapshot(self.snapshot_key, staged)
        finally:
            shutil.rmtree(staged, ignore_errors=True)
        self.project_dir = workspace.fetch_snapshot(self.snapshot_key)
        self.misen_requirement = _misen_bootstrap_requirement(
            self.project_dir,
            paths_visible=self.transport is None,
        )

        self.env_file_refs: list[str] = [
            workspace.put_job_file(self.submission_id, src.name, src.read_bytes())
            for src in _env_file_paths()
            if src.exists()
        ]

        self.prewarmed: _MaterializedEnvs | None = None
        store_root = _resolve_store_root(env_store_dir)
        if prewarm:
            self.prewarmed = _materialize_envs(self.project_dir, store_root, pixi_bin=self.pixi_bin)

    def prepare_job(
        self,
        work_unit: WorkUnit,
        workspace: Workspace,
        gpu_runtime: GpuRuntime,
        *,
        cpu_indices: list[int] | None,
        gpu_indices: list[int] | None,
    ) -> tuple[str, list[str], dict[str, str], Path]:
        """Prepare command/env overrides to execute one work unit.

        Prewarmed snapshots emit the worker command directly (activation
        paths are known); otherwise the command is a
        single Bash program that resolves tools, executes the workspace's
        transport when paths are not worker-visible, then invokes
        :mod:`misen.utils.materialize_env` with local paths.

        Args:
            work_unit: Work unit to execute.
            workspace: Workspace for payload/log storage.
            gpu_runtime: Runtime environment for GPU resources.
            cpu_indices: CPU logical-core indices for worker affinity, or
                ``None`` to leave inherited affinity untouched.
            gpu_indices: GPU device indices for worker visibility, or ``None``
                to leave inherited visibility untouched.

        Returns:
            Tuple ``(job_id, argv, env_overrides, log_path)``.

        Raises:
            RuntimeError: If this dispatch mode is unusable with the
                workspace (see messages).
        """
        job_id = token_base32(6)
        payload_ref = self.workspace.put_job_file(
            self.submission_id, f"{job_id}.pkl", work_unit.as_payload(workspace=workspace, job_id=job_id)
        )
        log_path = self.workspace.get_job_log(job_id=job_id, work_unit=work_unit)

        if self.prewarmed is not None:
            argv = _worker_command(
                self.prewarmed,
                [Path(ref) for ref in self.env_file_refs],
                Path(payload_ref),
                gpu_runtime,
                cpu_indices=cpu_indices,
                gpu_indices=gpu_indices,
                log_path=log_path,
            )
            return job_id, argv, _activation_env(self.prewarmed), log_path

        if self.misen_requirement is None:
            msg = (
                "The worker-side env bootstrap needs an installable misen: pin misen from a "
                "package index or Git commit in the project's uv.lock (a local misen checkout "
                "works only when the workspace serves files as shared paths), or use prewarm_envs."
            )
            raise RuntimeError(msg)
        transport = self.transport

        if transport is None:
            project_dir: Path | None = self.project_dir
            snapshot_key: str | None = None
            payload = payload_ref
            env_files = list(self.env_file_refs)
        else:
            project_dir = None
            snapshot_key = self.snapshot_key
            payload = payload_ref
            env_files = list(self.env_file_refs)

        worker_args = [
            "--gpu-runtime",
            gpu_runtime,
            *_indices_argv("cpu-indices", cpu_indices),
            *_indices_argv("gpu-indices", gpu_indices),
            JOB_LOG_PATH_ARG,
            str(log_path),
        ]
        script = worker_bootstrap_script(
            uv_bin=_uv_bin(),
            pixi_bin=self.pixi_bin,
            requires_pixi=self.pixi_bin is not None,
            transport_script=transport,
            misen_requirement=self.misen_requirement,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
            store_root=_resolve_store_root(self.env_store_dir),
            project_dir=project_dir,
            snapshot_key=snapshot_key,
            payload=payload,
            env_files=env_files,
            worker_args=worker_args,
        )
        return job_id, ["bash", "-c", script], {}, log_path

    def _stage(self, staged_dir: Path) -> None:
        """Stage code and dependency metadata for the workspace snapshot store.

        Raises:
            RuntimeError: If any uv invocation fails, or the project's pixi
                manifests fail validation.
        """
        packages_dir = staged_dir / _PACKAGES_DIR_NAME
        packages_dir.mkdir(parents=True)

        _run_tool([_uv_bin(), "lock"], error_msg="Lockfile resolution (uv lock) failed")

        for name in ("pyproject.toml", "uv.lock"):
            shutil.copy(Path.cwd() / name, staged_dir / name)
        python_version = Path.cwd() / ".python-version"
        if python_version.exists():
            shutil.copy(python_version, staged_dir / python_version.name)

        for package_path in _local_package_paths(Path.cwd() / "uv.lock"):
            if package_path.is_dir():
                _stage_local_package(package_path, packages_dir)
            else:
                # ``path`` dependencies that are already wheel/sdist files.
                shutil.copy(package_path, packages_dir / package_path.name)

        pixi_project = _resolve_project_pixi()
        if pixi_project is not None:
            manifest_path, lock_path, self.pixi_bin = pixi_project
            shutil.copy(manifest_path, staged_dir / "pixi.toml")
            shutil.copy(lock_path, staged_dir / "pixi.lock")


@contextmanager
def apply_env_files_temporarily() -> Iterator[None]:
    """Temporarily load environment variables from dotenv files.

    Later files override earlier ones. Modified keys are restored after exiting
    the context.
    """
    initial_environ = os.environ.copy()
    for f in _env_file_paths():
        if f.exists():
            load_dotenv(f, override=True)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(initial_environ)


def _env_file_paths() -> list[Path]:
    """Return ``.env`` / ``.env.local`` paths in the current working directory."""
    return [Path.cwd() / name for name in (".env", ".env.local")]


@cache
def _active_env_files() -> tuple[Path, ...]:
    """Return ``.env`` / ``.env.local`` paths that exist in CWD (cached)."""
    return tuple(p for p in _env_file_paths() if p.exists())


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

_ENV_STORE_SCHEMA = 1  # bump when an entry format changes incompatibly
_PACKAGES_DIR_NAME = "packages"  # staged local-package artifacts (wheels/sdists)
# Fixed build timestamp (1980-01-01, zip's minimum) so unchanged source
# rebuilds byte-identical artifacts and snapshot/overlay keys stay stable.
_SOURCE_DATE_EPOCH = "315532800"
# Multi-minute builds: the lifetime-minus-refresh headroom must exceed NFS
# attribute-cache staleness (~60s), or a waiter reading a stale lockfile
# mtime could break a live builder's lease.
_BUILD_LOCK_LIFETIME_S = 120
_BUILD_LOCK_REFRESH_S = 30


def _run_tool(
    argv: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None, error_msg: str
) -> subprocess.CompletedProcess[str]:
    """Run a CLI tool, wrapping failures in a ``RuntimeError`` with its output."""
    try:
        return subprocess.run(argv, check=True, capture_output=True, text=True, env=env, cwd=cwd)  # noqa: S603
    except subprocess.CalledProcessError as e:
        msg = f"{error_msg}: {(e.stderr or e.stdout or '').strip()}"
        raise RuntimeError(msg) from None


def _prepend_path_var(head: str, existing: str | None) -> str:
    """Prepend ``head`` to a ``os.pathsep``-separated env value."""
    return head if not existing else f"{head}{os.pathsep}{existing}"


def _store_key(parts: tuple[object, ...]) -> str:
    """Hash a tuple of key components to an unpadded-base32 store key."""
    return Hash(hash_values(parts)).b32()


def _snapshot_key(staged_dir: Path) -> str:
    """Content key of a staged snapshot: every file's relative path and bytes.

    Package artifacts build under a pinned ``SOURCE_DATE_EPOCH``.
    Non-reproducible build backends only cost key stability (one snapshot
    entry per submission), never correctness.
    """
    files = sorted(p for p in staged_dir.rglob("*") if p.is_file())
    return _store_key((_ENV_STORE_SCHEMA, tuple((p.relative_to(staged_dir).as_posix(), p.read_bytes()) for p in files)))


def _is_pure_wheel(wheel_path: Path) -> bool:
    """Whether a wheel is platform-independent (``…-none-any.whl``)."""
    tags = wheel_path.stem.split("-")
    return len(tags) >= 5 and tags[-2:] == ["none", "any"]  # noqa: PLR2004


def _stage_local_package(package_dir: Path, packages_dir: Path) -> None:
    """Build one local package and stage its distributable artifact.

    Pure-python wheels are staged as wheels (fast worker installs, no
    build step at runtime). Native packages are staged as *sdists*: their
    wheels would bind this host's platform and toolchain, so they compile
    on the execution host instead, inside the pixi activation where the
    locked toolchain exists. When even the wheel build fails here (no
    local toolchain), the sdist is staged with a warning — buildability
    is then only verified at env-materialization time (prewarm restores
    fail-fast).

    Raises:
        RuntimeError: If no distributable artifact can be produced.
    """
    # Outside the staged tree so build residue can never leak into the
    # snapshot's content key.
    out_dir = Path(tempfile.mkdtemp(prefix="misen-pkg-build-"))
    build_env = os.environ.copy() | {"SOURCE_DATE_EPOCH": _SOURCE_DATE_EPOCH}
    try:
        try:
            _run_tool(
                [_uv_bin(), "build", "--out-dir", str(out_dir), str(package_dir)],
                env=build_env,
                error_msg=f"Build failed for local package {package_dir}",
            )
        except RuntimeError as e:
            logger.warning(
                "Wheel build failed for %s; staging its sdist only, which will build on the "
                "execution hosts (use prewarm_envs to verify buildability at submission). %s",
                package_dir,
                e,
            )
            _run_tool(
                [_uv_bin(), "build", "--sdist", "--out-dir", str(out_dir), str(package_dir)],
                env=build_env,
                error_msg=f"Source distribution build failed for local package {package_dir}",
            )
        wheels = sorted(out_dir.glob("*.whl"))
        sdists = sorted(out_dir.glob("*.tar.gz"))
        if wheels and _is_pure_wheel(wheels[-1]):
            artifact = wheels[-1]
        elif sdists:
            artifact = sdists[-1]
            if wheels:
                logger.debug("Staging sdist for platform-specific package %s.", package_dir)
        else:
            msg = f"No distributable artifact produced for local package {package_dir}."
            raise RuntimeError(msg)
        shutil.copy(artifact, packages_dir / artifact.name)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _misen_bootstrap_requirement(project_dir: Path, *, paths_visible: bool) -> str | None:
    """Resolve how a worker bootstrap installs misen, from the staged lock.

    A registry pin installs from the index (``misen==<version>``); a Git
    source becomes an immutable PEP 508 direct reference at the commit
    recorded in ``uv.lock``. A local misen checkout (developing misen itself)
    uses the artifact staged in ``packages/`` — usable only when the workspace
    serves files as shared paths. Returns ``None`` when no usable form exists;
    prewarmed snapshots never need one.
    """
    lock = tomllib.loads((project_dir / "uv.lock").read_text())
    for package in lock.get("package", []):
        if package.get("name") != "misen":
            continue
        source = package.get("source", {})
        if "registry" in source and "version" in package:
            return f"misen=={package['version']}"
        if isinstance(locked_git := source.get("git"), str):
            # uv records selectors in the query and the resolved commit in
            # the fragment, e.g. ``...?branch=main#<sha>``. Workers must use
            # the commit, never re-resolve the moving selector.
            parsed = urlsplit(locked_git)
            commit = parsed.fragment
            if commit:
                subdirectory = next(
                    (value for key, value in parse_qsl(parsed.query) if key == "subdirectory"),
                    None,
                )
                git_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
                if not git_url.startswith("git+"):
                    git_url = f"git+{git_url}"
                requirement = f"misen @ {git_url}@{commit}"
                if subdirectory:
                    requirement += f"#subdirectory={quote(subdirectory, safe='/')}"
                return requirement
        if any(kind in source for kind in ("editable", "directory", "path")) and paths_visible:
            artifacts = sorted(
                p for p in (project_dir / _PACKAGES_DIR_NAME).glob("misen-*") if p.name.split("-")[0] == "misen"
            )
            if artifacts:
                return str(artifacts[0])
        return None
    return None


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
    after ``build`` succeeds and only while the lock lease is still held.
    Failed builds are removed before releasing an owned lock; an entry without
    a marker is therefore crashed-builder or lost-lease residue and is safe to
    remove on the next attempt. ``sanity_path`` (entry-relative) must exist for
    a marked entry to count as usable, healing a durable marker that outlived
    its entry.

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
        try:
            build(entry_dir)
        except BaseException:
            # A normal failed builder still owns the entry and should not
            # strand a potentially huge partial environment. If the lease was
            # stolen, a replacement may already be writing this path, so only
            # that builder may clean it.
            if lock.is_locked() and entry_dir.exists():
                logger.warning("Removing failed %s build at %s.", label, entry_dir)
                try:
                    _rmtree_with_retry(entry_dir)
                except OSError:
                    logger.exception("Could not remove failed %s build at %s.", label, entry_dir)
            raise
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


def _venv_site_dir(venv_dir: Path) -> Path:
    """Return the single ``site-packages`` directory of a venv."""
    return next((venv_dir / "lib").glob("python*")) / "site-packages"


def _build_overlay_venv(
    venv_dir: Path, deps_env_dir: Path, packages: list[Path], *, install_prefix: list[str] | None = None
) -> Path:
    """Build an overlay venv chained to ``deps_env_dir`` and install ``packages``.

    ``packages`` are the staged local-package artifacts. ``install_prefix``
    (the pixi activation wrap, when the project has a conda env) prefixes
    the install command so sdists compile against the locked native
    toolchain. Returns the overlay's site-packages directory.

    Raises:
        RuntimeError: If any uv invocation fails.
    """
    _run_tool(
        # ``--python <deps env python>`` resolves to its *base* interpreter.
        [_uv_bin(), "venv", str(venv_dir), "--python", str(deps_env_dir / "bin" / "python")],
        error_msg="Overlay virtual environment creation failed",
    )
    overlay_site = _venv_site_dir(venv_dir)
    deps_site = deps_env_dir / overlay_site.relative_to(venv_dir)
    (overlay_site / "_misen_shared_env.pth").write_text(f"{deps_site}\n")

    if packages:
        _run_tool(
            [
                *(install_prefix or []),
                _uv_bin(),
                "pip",
                "install",
                "--python",
                str(venv_dir / "bin" / "python"),
                "--no-deps",
                "--no-cache",
                "--compile-bytecode",
                *(str(p) for p in packages),
            ],
            error_msg="Local package installation failed",
        )
    return overlay_site


# --------------------------------------------------------------------------
# Env materialization: build (or reuse) the environments described by a
# published snapshot, in a content-keyed env store on this host. Same store
# protocol as above; runs on the submitting host (prewarm) or on execution
# hosts via the submitted Bash bootstrap and ``misen.utils.materialize_env``.
# --------------------------------------------------------------------------


class _MaterializedEnvs(NamedTuple):
    """Env-store paths a job activates, produced by :func:`_materialize_envs`."""

    deps_env_dir: Path
    overlay_venv_dir: Path
    overlay_site_dir: Path
    conda_manifest_path: Path | None
    pixi_bin: str | None


def _default_runtime_store_root() -> Path:
    """Default env-store root on the building host.

    Deliberately a fixed per-user path under ``/tmp`` (not
    ``tempfile.gettempdir()``): schedulers like SLURM commonly export a
    per-job ``TMPDIR``, which would defeat cross-job reuse of the store on
    a node. Sites where ``/tmp`` is small or RAM-backed should point
    ``env_store_dir`` at persistent node-local scratch instead.
    """
    try:
        user = getpass.getuser()
    except (KeyError, OSError):
        user = f"uid{os.getuid()}"
    return Path("/tmp") / f"misen-env-store-{user}"  # noqa: S108


def _resolve_store_root(env_store_dir: str | Path | None) -> Path:
    """Resolve the configured env-store root, shared by submitter and worker.

    ``absolute()`` (not ``resolve()``): build steps run under differing
    CWDs so relative paths must pin down, but symlink resolution would
    give one host's spelling of a path that other hosts name differently,
    splitting the store.
    """
    return Path(env_store_dir).absolute() if env_store_dir is not None else _default_runtime_store_root()


def _worker_command(
    envs: _MaterializedEnvs,
    env_files: list[Path],
    payload_path: Path,
    gpu_runtime: GpuRuntime,
    *,
    cpu_indices: list[int] | None,
    gpu_indices: list[int] | None,
    log_path: Path,
) -> list[str]:
    """Build the worker command for materialized envs.

    The single definition of what a job runs — the pixi activation wrap
    (when the snapshot has a conda env) around the ``uv run … -m
    misen.utils.execute`` invocation — used identically by prewarmed
    dispatch (at submission) and the bootstrap (on the execution host).
    """
    argv: list[str] = []
    if envs.conda_manifest_path is not None and envs.pixi_bin is not None:
        argv += _pixi_run_prefix(envs.pixi_bin, envs.conda_manifest_path)
    argv += _uv_execute_argv(env_files, payload_path, gpu_runtime, cpu_indices=cpu_indices, gpu_indices=gpu_indices)
    argv += [JOB_LOG_PATH_ARG, str(log_path)]
    return argv


def _activation_env(envs: _MaterializedEnvs) -> dict[str, str]:
    """Env overrides activating materialized envs; pairs with :func:`_worker_command`.

    ``uv run`` prepends only the overlay venv's bin; the deps env's bin
    carries dependency console scripts (e.g. torchrun), and PYTHONPATH
    keeps local packages importable for children of deps-env script
    shebangs (which never read the overlay's ``.pth``).
    """
    return {
        "VIRTUAL_ENV": str(envs.overlay_venv_dir),
        "PATH": _prepend_path_var(str(envs.deps_env_dir / "bin"), os.environ.get("PATH")),
        "PYTHONPATH": _prepend_path_var(str(envs.overlay_site_dir), os.environ.get("PYTHONPATH")),
    }


def _resolve_pixi_bin(preferred: str | None) -> str:
    """Return a usable pixi CLI path on this host.

    Prefers the path recorded at staging time (typically valid on the
    execution hosts too) and falls back to a PATH lookup.

    Raises:
        RuntimeError: If no usable ``pixi`` CLI is found.
    """
    if preferred is not None and os.access(preferred, os.X_OK):
        return preferred
    pixi_bin = shutil.which("pixi")
    if pixi_bin is None:
        msg = (
            "This snapshot has a conda env, but no usable `pixi` CLI was found on this "
            "host. Install it from https://pixi.sh or make it visible on PATH."
        )
        raise RuntimeError(msg)
    return pixi_bin


def _materialize_envs(project_dir: Path, store_root: Path, *, pixi_bin: str | None = None) -> _MaterializedEnvs:
    """Ensure this host's env-store entries for a published snapshot.

    The conda env comes first: staged sdists compile inside the pixi
    activation, so the overlay build needs the conda entry available.

    Raises:
        RuntimeError: If any build fails or a build lock is lost.
    """
    conda_manifest_path: Path | None = None
    resolved_pixi: str | None = None
    if (project_dir / "pixi.lock").exists():
        resolved_pixi = _resolve_pixi_bin(pixi_bin)
        conda_manifest_path = _ensure_conda_env_entry(
            manifest_path=project_dir / "pixi.toml",
            lock_path=project_dir / "pixi.lock",
            pixi_bin=resolved_pixi,
            store_root=store_root,
        )

    package_install_prefix = (
        _pixi_run_prefix(resolved_pixi, conda_manifest_path)
        if resolved_pixi is not None and conda_manifest_path is not None
        else None
    )
    deps_env_dir, overlay_venv_dir, overlay_site_dir = _ensure_python_env(
        project_dir, store_root, package_install_prefix=package_install_prefix
    )
    return _MaterializedEnvs(deps_env_dir, overlay_venv_dir, overlay_site_dir, conda_manifest_path, resolved_pixi)


def _python_env_key(project_dir: Path) -> str:
    """Content key of the deps env for a published snapshot on this host.

    Captures the frozen uv project, interpreter-selection inputs
    (``.python-version`` pin and ``UV_PYTHON``), and this host's platform.
    The project metadata is part of the key because it controls dependency
    groups, sources, indexes, and build settings that ``uv sync`` applies.
    """
    pin = project_dir / ".python-version"
    return _store_key(
        (
            _ENV_STORE_SCHEMA,
            (project_dir / "pyproject.toml").read_bytes(),
            (project_dir / "uv.lock").read_bytes(),
            pin.read_bytes() if pin.exists() else None,
            os.environ.get("UV_PYTHON"),
            sys.platform,
            platform.machine(),
        )
    )


def _ensure_python_env(
    project_dir: Path, store_root: Path, *, package_install_prefix: list[str] | None = None
) -> tuple[Path, Path, Path]:
    """Materialize the python env for a published snapshot on this host.

    Two store entries: a dependency env produced by frozen ``uv sync`` with
    every local package excluded, and an overlay containing the exact local
    artifacts staged at submission. The dependency env therefore preserves
    uv's native lock, source, index, and build semantics while remaining
    reusable across code-only changes.

    Returns:
        Tuple ``(deps_env_dir, overlay_venv_dir, overlay_site_dir)``.

    Raises:
        RuntimeError: If any uv invocation fails or a build lock is lost.
    """
    deps_key = _python_env_key(project_dir)

    def build_deps(env_dir: Path) -> None:
        _run_tool(
            [
                _uv_bin(),
                "sync",
                "--frozen",
                "--no-install-local",
                "--compile-bytecode",
                "--project",
                str(project_dir),
            ],
            # An explicit project-environment path keeps the immutable
            # snapshot untouched and builds directly at the final venv path.
            # Cache policy stays lazy, so warm store reuse invokes no uv
            # subprocess at all.
            env=os.environ.copy() | _uv_cache_env(store_root) | {"UV_PROJECT_ENVIRONMENT": str(env_dir)},
            cwd=project_dir,
            error_msg="Dependency environment sync failed",
        )

    deps_env_dir = _ensure_store_entry(
        store=store_root / "python-envs",
        key=deps_key,
        build=build_deps,
        sanity_path="bin/python",
        label="python env",
    )

    packages = sorted(p for p in (project_dir / _PACKAGES_DIR_NAME).iterdir() if p.is_file())
    overlay_key = _store_key(
        (_ENV_STORE_SCHEMA, deps_key, tuple((package.name, package.read_bytes()) for package in packages))
    )

    def build_overlay(entry_dir: Path) -> None:
        entry_dir.mkdir(parents=True)
        _build_overlay_venv(entry_dir / "venv", deps_env_dir, packages, install_prefix=package_install_prefix)

    overlay_entry = _ensure_store_entry(
        store=store_root / "overlay-envs",
        key=overlay_key,
        build=build_overlay,
        sanity_path="venv/bin/python",
        label="overlay env",
    )
    overlay_venv_dir = overlay_entry / "venv"
    return deps_env_dir, overlay_venv_dir, _venv_site_dir(overlay_venv_dir)


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


def _resolve_project_pixi() -> tuple[Path, Path, str] | None:
    """Locate and validate the caller's pixi manifests.

    Returns:
        Tuple ``(manifest_path, lock_path, pixi_bin)``, or ``None`` when no
        ``pixi.lock`` is in CWD.

    Raises:
        RuntimeError: If ``pixi.lock`` has no adjacent ``pixi.toml``, the
            lockfile references PyPI packages, or the ``pixi`` CLI is
            missing.
    """
    lock_path = Path.cwd() / "pixi.lock"
    if not lock_path.exists():
        return None

    manifest_path = lock_path.parent / "pixi.toml"
    if not manifest_path.exists():
        msg = f"Found {lock_path.name} but no pixi.toml next to it."
        raise RuntimeError(msg)

    _check_pixi_lock_for_pypi(lock_path)

    pixi_bin = shutil.which("pixi")
    if pixi_bin is None:
        msg = (
            "A pixi.lock was detected but the `pixi` CLI is not on PATH. "
            "Install it from https://pixi.sh to use conda dependencies with misen."
        )
        raise RuntimeError(msg)
    return manifest_path, lock_path, pixi_bin


def _ensure_conda_env_entry(*, manifest_path: Path, lock_path: Path, pixi_bin: str, store_root: Path) -> Path:
    """Build-or-reuse the conda env store entry for a pixi manifest pair.

    The entry holds copies of both manifests plus the installed
    ``.pixi/envs/default`` prefix, keyed by the manifests' bytes and the
    building platform. The cache-dir policy (a ``pixi info`` subprocess)
    is resolved lazily inside the build so the warm reuse path pays
    nothing.

    Returns:
        Path to the entry's ``pixi.toml`` (pixi's manifest-path flag
        consumes this).

    Raises:
        RuntimeError: If ``pixi install`` fails or the build lock is lost.
    """
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
            env=os.environ.copy() | _pixi_cache_env(store_root),
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
    ``env_files`` source (live CWD paths for :func:`prepare_live_job`,
    submission-scoped job files for :class:`ProjectSnapshot`).
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
    """Render ``--<flag> i j k``, or nothing for ``None`` / ``[]``."""
    if indices is None or len(indices) == 0:
        return []
    return [f"--{flag}", *(str(i) for i in indices)]


def _pixi_run_prefix(pixi_bin: str, manifest_path: Path) -> list[str]:
    """Return a ``pixi run --frozen -- …`` activation wrapper.

    ``--`` stops pixi from parsing the wrapped executable's flags and works
    across both older and current Pixi releases.
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
        "--",
    ]


@cache
def _detect_pixi_wrap() -> list[str]:
    """Return a ``pixi run`` argv prefix for in-tree activation, or ``[]``.

    Cached on the process — CWD and ``pixi.toml`` are assumed stable for
    misen's lifetime. Used by :func:`prepare_live_job` so live dispatch
    still gets conda activation when the caller's project declares a pixi
    env. Unlike :class:`ProjectSnapshot`, which stages ``pixi.toml`` /
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
