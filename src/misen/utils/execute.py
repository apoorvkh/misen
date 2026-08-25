"""CLI entrypoint for executing serialized work-unit payloads.

The worker process runs this module to execute a single work unit.  It also
owns the **job-log lifecycle**: the parent (executor) tells the worker where to
write its log via ``--job-log-path``; the worker wraps its full lifecycle in
``workspace.streaming_job_log(...)`` so a remote workspace can publish the
local file to shared storage as it grows.  This way the same code captures
every byte the worker emits -- allocation/setup, ``WorkUnit.execute``,
post-execute finalizers -- regardless of where the worker is running.
"""

import contextlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cloudpickle
import tyro
from dotenv import load_dotenv

from misen.utils.dask_runtime import run_role_from_env

if TYPE_CHECKING:
    from collections.abc import Callable

_ENV_FILES_LOADED = "MISEN_ENV_FILES_LOADED"


def execute(
    payload: Path,
    *,
    env_file: tuple[Path, ...] = (),
    job_log_path: Path | None = None,
) -> None:
    """Execute a cloudpickle work-unit payload file.

    Args:
        payload: Path to payload bytes; the payload is a dict with keys
            ``workspace`` (Workspace) and ``fn`` (zero-arg callable). The
            workspace is exposed so the worker can wrap its lifecycle in
            :meth:`Workspace.streaming_job_log` for live log publishing.
        env_file: Dotenv files loaded in order. Later files override earlier
            files, while variables already present in the worker environment win.
        job_log_path: Path where the parent executor is writing this worker's
            combined stdout/stderr log. When provided, the workspace can stream
            the log while the worker is still running.
    """
    env_files_loaded = os.environ.pop(_ENV_FILES_LOADED, None)
    reload_with_env = bool(env_file) and env_files_loaded is None
    if reload_with_env:
        inherited_env = os.environ.copy()
        for path in env_file:
            load_dotenv(path, override=True)
        os.environ.update(inherited_env)
    if virtual_env := os.environ.get("VIRTUAL_ENV"):
        venv_bin = str(Path(virtual_env) / "bin")
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        os.environ["PATH"] = os.pathsep.join((venv_bin, *(path for path in path_entries if path != venv_bin)))
    if reload_with_env:
        env = os.environ.copy()
        env[_ENV_FILES_LOADED] = "1"
        sys.stdout.flush()
        sys.stderr.flush()
        os.execve(sys.executable, [sys.executable, "-m", "misen.utils.execute", *sys.argv[1:]], env)  # noqa: S606
        return

    if run_role_from_env():
        return

    bundle = cloudpickle.loads(payload.read_bytes())
    workspace = bundle["workspace"]
    payload_fn: Callable[[], None] = bundle["fn"]

    # The parent points the executor's stdout (a direct processkit redirect /
    # SLURM ``--output=...``) at this same path, so the live uploader
    # sees everything the worker writes -- allocation/setup,
    # ``WorkUnit.execute``, post-execute Python finalizers.
    streaming = workspace.streaming_job_log(job_log_path) if job_log_path is not None else contextlib.nullcontext()

    with streaming:
        payload_fn()


if __name__ == "__main__":
    tyro.cli(execute)
