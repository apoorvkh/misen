"""Materialize an already-local project snapshot and execute its payload."""

# No ``from __future__ import annotations``: tyro evaluates annotations at runtime.
import os
import shutil
import sys
from pathlib import Path

import tyro

from misen.task_metadata import GpuRuntime
from misen.utils.bootstrap_transport import PIXI_BIN_ENV, UV_BIN_ENV
from misen.utils.snapshot import (
    _activation_env,
    _materialize_envs,
    _publish_marker,
    _resolve_store_root,
    _snapshot_key,
    _worker_command,
)


def main(
    *,
    project_dir: Path,
    payload: Path,
    gpu_runtime: GpuRuntime,
    job_log_path: Path,
    env_file: tuple[Path, ...] = (),
    snapshot_key: str | None = None,
    env_store_root: Path | None = None,
    pixi_bin: str | None = None,
    cpu_indices: list[int] | None = None,
    gpu_indices: list[int] | None = None,
) -> None:
    """Build/reuse environments for local snapshot data, then exec the job.

    ``snapshot_key`` is supplied for shell-transported snapshots. The first
    consumer verifies the fetched tree before publishing a completion marker;
    later jobs reuse the verified tree without rehashing it.
    """
    project_dir = project_dir.absolute()
    store_root = _resolve_store_root(env_store_root)
    if snapshot_key is not None:
        expected_project_dir = (store_root / "snapshots" / snapshot_key).absolute()
        if project_dir != expected_project_dir:
            msg = f"Transported snapshot must be materialized at {expected_project_dir}, not {project_dir}."
            raise ValueError(msg)
        marker = project_dir.parent / f"{snapshot_key}.complete"
        if not marker.is_file():
            actual_key = _snapshot_key(project_dir)
            if actual_key != snapshot_key:
                if project_dir.is_symlink():
                    project_dir.unlink()
                else:
                    shutil.rmtree(project_dir)
                msg = f"Transported snapshot has content key {actual_key}, expected {snapshot_key}."
                raise RuntimeError(msg)
            _publish_marker(project_dir.parent, marker)

    envs = _materialize_envs(project_dir, store_root, pixi_bin=pixi_bin)
    command = _worker_command(
        envs,
        list(env_file),
        payload,
        gpu_runtime,
        cpu_indices=cpu_indices,
        gpu_indices=gpu_indices,
        log_path=job_log_path,
    )

    env = os.environ.copy()
    for name in (UV_BIN_ENV, PIXI_BIN_ENV):
        env.pop(name, None)
    env.update(_activation_env(envs))

    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(command[0], command, env)  # noqa: S606


if __name__ == "__main__":
    tyro.cli(main)
