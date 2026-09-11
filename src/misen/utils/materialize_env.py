"""Materialize an already-local project snapshot and execute its payload."""

# No ``from __future__ import annotations``: tyro evaluates annotations at runtime.
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path

import tyro

from misen.exceptions import SnapshotError
from misen.utils.bootstrap_transport import PIXI_BIN_ENV
from misen.utils.snapshot import (
    _activation_env,
    _materialize_envs,
    _publish_marker,
    _resolve_store_root,
    _snapshot_key,
    _worker_command,
)
from misen.utils.uv_tool import UV_BIN_ENV


def main(
    *,
    project_dir: Path,
    payload: Path,
    job_log_path: Path,
    env_file: tuple[Path, ...] = (),
    snapshot_key: str | None = None,
    env_store_root: Path | None = None,
    pixi_bin: str | None = None,
    prepared_command: Path | None = None,
) -> None:
    """Build/reuse environments for local snapshot data, then exec the job.

    ``snapshot_key`` is supplied for shell-transported snapshots. The first
    consumer verifies the fetched tree before publishing a completion marker;
    later jobs reuse the verified tree without rehashing it.

    Raises:
        ValueError: If a transported snapshot is materialized at the wrong
            path for its content key.
        SnapshotError: If snapshot verification or environment materialization
            fails.
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
            try:
                actual_key = _snapshot_key(project_dir)
                if actual_key != snapshot_key:
                    if project_dir.is_symlink():
                        project_dir.unlink()
                    else:
                        shutil.rmtree(project_dir)
                    msg = f"Transported snapshot has content key {actual_key}, expected {snapshot_key}."
                    raise SnapshotError(msg)
                _publish_marker(project_dir.parent, marker, target=project_dir)
            except OSError as exc:
                msg = f"Could not verify transported snapshot {snapshot_key}: {exc}"
                raise SnapshotError(msg) from exc

    envs = _materialize_envs(project_dir, store_root, pixi_bin=pixi_bin)
    if prepared_command is not None:
        # Keep activation paths, never a preparation job's resource/env overrides.
        argv = _worker_command(envs, [], Path("__MISEN_PAYLOAD__"), log_path=Path("__MISEN_LOG__"))
        command = shlex.join(argv[:-3]) + ' "$@"'
        activation = (
            f"export VIRTUAL_ENV={shlex.quote(str(envs.overlay_venv_dir))}\n"
            f"export PATH={shlex.quote(str(envs.deps_env_dir / 'bin'))}${{PATH:+:$PATH}}\n"
            f"export PYTHONPATH={shlex.quote(str(envs.overlay_site_dir))}${{PYTHONPATH:+:$PYTHONPATH}}\n"
        )
        prepared_command.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=prepared_command.parent, delete=False) as output:
            output.write(f"set -eu\nunset {UV_BIN_ENV} {PIXI_BIN_ENV}\n{activation}exec {command}\n")
        Path(output.name).replace(prepared_command)
    if os.environ.pop("MISEN_PREPARE_ONLY", ""):
        return
    command = _worker_command(
        envs,
        list(env_file),
        payload,
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
