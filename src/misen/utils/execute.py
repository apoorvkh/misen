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
from pathlib import Path
from typing import TYPE_CHECKING

import cloudpickle
import tyro

from misen.task_metadata import AcceleratorType
from misen.utils.dask_runtime import run_role_from_env
from misen.utils.resource_binding import apply_resource_binding

if TYPE_CHECKING:
    from collections.abc import Callable


def execute(
    payload: Path,
    *,
    cpu_indices: list[int] | None = None,
    accelerator_type: AcceleratorType = "cuda",
    accelerator_indices: list[int] | None = None,
    job_log_path: Path | None = None,
) -> None:
    """Execute a cloudpickle work-unit payload file.

    Args:
        payload: Path to payload bytes; the payload is a dict with keys
            ``workspace`` (Workspace) and ``fn`` (zero-arg callable). The
            workspace is exposed so the worker can wrap its lifecycle in
            :meth:`Workspace.streaming_job_log` for live log publishing.
        cpu_indices: CPU logical-core indices to bind via
            :func:`os.sched_setaffinity`. Pass ``None`` when the scheduler
            (e.g. SLURM) already pins CPUs for this process.
        accelerator_type: Accelerator backend whose visibility should be bound.
        accelerator_indices: Device indices assigned by a host-level executor,
            ``[]`` to hide all maskable devices, or ``None`` to preserve
            scheduler-provided isolation.
        job_log_path: Path where the parent executor is writing this worker's
            combined stdout/stderr log. When provided, the workspace can stream
            the log while the worker is still running.
    """
    apply_resource_binding(
        cpu_indices=cpu_indices,
        accelerator_type=accelerator_type,
        accelerator_indices=accelerator_indices,
    )
    if run_role_from_env():
        return

    bundle = cloudpickle.loads(payload.read_bytes())
    workspace = bundle["workspace"]
    payload_fn: Callable[[], None] = bundle["fn"]

    # The parent points the scheduler's stdout (``Popen(stdout=...)`` /
    # SLURM ``--output=...``) at this same path, so the live uploader
    # sees everything the worker writes -- allocation/setup,
    # ``WorkUnit.execute``, post-execute Python finalizers.
    streaming = workspace.streaming_job_log(job_log_path) if job_log_path is not None else contextlib.nullcontext()

    with streaming:
        payload_fn()


if __name__ == "__main__":
    tyro.cli(execute)
