"""Worker-side entrypoint that materializes a snapshot's environments.

Non-prewarmed jobs are dispatched as ``uv run --no-project --with misen…
-m misen.utils.bootstrap_env …``: uv provisions a minimal ephemeral env
holding misen (and its dependencies) so this module can run on an execution
host that shares nothing with the submitter but the workspace's storage.

The bootstrap deliberately never reconstructs a ``Workspace`` (a custom
subclass would be unimportable here) and never unpickles anything. It
consumes only a **data plane**, in one of two forms chosen by the
workspace's :meth:`~misen.workspace.Workspace.bootstrap_transport`:

- ``path``: the snapshot dir, payload, and env files arrive as plain
  worker-visible paths in argv — no transport machinery at all;
- ``obstore``: argv carries content keys/refs and the
  ``MISEN_BOOTSTRAP_TRANSPORT`` env var carries an object-store
  description; blobs are fetched with the same misen-built-in helpers
  ``CloudWorkspace`` itself uses, into the env-store root (snapshots are
  content-addressed there, so jobs landing on one host fetch once).

It then materializes the conda/python envs into the content-keyed env
store on this host's disk (concurrent jobs coordinate through the store's
build locks), applies the same activation a prewarmed dispatch encodes at
submission time (``VIRTUAL_ENV`` / ``PATH`` / ``PYTHONPATH`` overrides and
the pixi wrap), and replaces itself with the worker command via ``exec``.

Anything this module prints lands in the job log (the executor already
redirects the job's stdout/stderr before the worker starts). The payload
itself is only unpickled later, inside the project env, by
``misen.utils.execute`` — where any custom workspace's library is
guaranteed present, because the user's project depends on it.
"""

# No ``from __future__ import annotations``: tyro builds the CLI from
# runtime-evaluated annotations (same as ``misen.utils.execute``).
import json
import os
import sys
from pathlib import Path

import tyro

from misen.task_metadata import GpuRuntime
from misen.utils.snapshot import (
    BOOTSTRAP_TRANSPORT_ENV,
    _activation_env,
    _materialize_envs,
    _resolve_store_root,
    _worker_command,
)


def main(
    *,
    gpu_runtime: GpuRuntime,
    job_log_path: Path,
    project_dir: Path | None = None,
    payload: Path | None = None,
    env_file: tuple[Path, ...] = (),
    snapshot_key: str | None = None,
    payload_ref: str | None = None,
    env_file_ref: tuple[str, ...] = (),
    env_store_root: Path | None = None,
    pixi_bin: str | None = None,
    cpu_indices: list[int] | None = None,
    gpu_indices: list[int] | None = None,
) -> None:
    """Materialize this host's envs for a published snapshot, then exec the job.

    Exactly one data plane must be given: paths (``project_dir`` +
    ``payload`` [+ ``env_file``]) or refs (``snapshot_key`` +
    ``payload_ref`` [+ ``env_file_ref``], with the transport in the
    ``MISEN_BOOTSTRAP_TRANSPORT`` env var).

    Args:
        gpu_runtime: GPU runtime for the worker (e.g. ``cuda``).
        job_log_path: Job log path for the worker lifecycle.
        project_dir: Snapshot directory (path transport).
        payload: Serialized payload path (path transport).
        env_file: Env-file paths (path transport).
        snapshot_key: Snapshot content key (ref transport).
        payload_ref: Payload blob ref (ref transport).
        env_file_ref: Env-file blob refs (ref transport).
        env_store_root: Env-store root on this host's disk.
        pixi_bin: Preferred pixi CLI path (falls back to PATH lookup).
        cpu_indices: CPU indices to bind in the worker.
        gpu_indices: GPU indices to mask in the worker.

    Raises:
        RuntimeError: If the data-plane arguments are incomplete, the
            transport env var is missing/unsupported for ref mode, or the
            snapshot needs a conda env but no usable ``pixi`` CLI exists
            on this host.
    """
    store_root = _resolve_store_root(env_store_root)

    if project_dir is not None:
        if payload is None:
            msg = "--payload is required with --project-dir"
            raise RuntimeError(msg)
        project_dir = project_dir.absolute()
        payload_path = payload
        env_files = list(env_file)
    elif snapshot_key is not None:
        if payload_ref is None:
            msg = "--payload-ref is required with --snapshot-key"
            raise RuntimeError(msg)
        project_dir, payload_path, env_files = _fetch_data_plane(
            store_root, snapshot_key, payload_ref, list(env_file_ref)
        )
    else:
        msg = "either --project-dir/--payload or --snapshot-key/--payload-ref is required"
        raise RuntimeError(msg)

    envs = _materialize_envs(project_dir, store_root, pixi_bin=pixi_bin)
    command = _worker_command(
        envs,
        env_files,
        payload_path,
        gpu_runtime,
        cpu_indices=cpu_indices,
        gpu_indices=gpu_indices,
        log_path=job_log_path,
    )

    # The exact activation a prewarmed dispatch encodes at submission time.
    env = os.environ.copy()
    env.pop(BOOTSTRAP_TRANSPORT_ENV, None)
    env.update(_activation_env(envs))

    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(command[0], command, env)  # noqa: S606


def _fetch_data_plane(
    store_root: Path, snapshot_key: str, payload_ref: str, env_file_refs: list[str]
) -> tuple[Path, Path, list[Path]]:
    """Fetch the snapshot, payload, and env files via the declared transport.

    Blobs land under the env-store root: snapshots are content-addressed
    (fetched once per host, shared by every job of the same code state),
    job files under their refs' natural relative layout.

    Raises:
        RuntimeError: If the transport env var is missing or its kind is
            not supported by this misen build.
        FileNotFoundError: If a referenced blob does not exist.
    """
    raw = os.environ.get(BOOTSTRAP_TRANSPORT_ENV)
    if not raw:
        msg = f"{BOOTSTRAP_TRANSPORT_ENV} must carry the data-plane transport for ref-based bootstraps."
        raise RuntimeError(msg)
    transport = json.loads(raw)
    if transport.get("kind") != "obstore":
        msg = f"Unsupported bootstrap transport kind: {transport.get('kind')!r}"
        raise RuntimeError(msg)

    # Deferred import: path-transport bootstraps never touch obstore.
    from misen.workspaces.cloud import _build_obstore_store, _download_file, _download_snapshot

    store = _build_obstore_store(
        transport["backend"],
        transport["bucket"],
        endpoint=transport.get("endpoint"),
        s3_region=transport.get("s3_region"),
        config=transport.get("config") or {},
    )
    prefix = transport.get("prefix") or ""

    project_dir = _download_snapshot(store, prefix, snapshot_key, store_root / "snapshots" / snapshot_key)
    payload_path = _download_file(store, payload_ref, _job_file_target(store_root, payload_ref))
    env_files = [_download_file(store, ref, _job_file_target(store_root, ref)) for ref in env_file_refs]
    return project_dir, payload_path, env_files


def _job_file_target(store_root: Path, ref: str) -> Path:
    """Local path for a fetched job-file ref (refs are trusted object keys)."""
    parts = [p for p in ref.split("/") if p not in ("", ".", "..")]
    return store_root.joinpath("job-files", *parts)


if __name__ == "__main__":
    tyro.cli(main)
