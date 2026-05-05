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

from misen.task_metadata import GpuRuntime
from misen.utils.resource_binding import apply_resource_binding

if TYPE_CHECKING:
    from collections.abc import Callable


def execute(
    payload: Path,
    *,
    cpu_indices: list[int] | None = None,
    gpu_indices: list[int] | None = None,
    gpu_runtime: GpuRuntime = "cuda",
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
        gpu_indices: GPU device indices to mask via the runtime's visibility
            environment variables. Pass ``None`` when the scheduler already
            masks GPUs (e.g. SLURM cgroups).
        gpu_runtime: Runtime selecting which visibility env vars to set.
        job_log_path: Path where the parent executor is writing this worker's
            combined stdout/stderr log. When provided, the workspace can stream
            the log while the worker is still running.
    """
    apply_resource_binding(
        cpu_indices=cpu_indices,
        gpu_indices=gpu_indices,
        gpu_runtime=gpu_runtime,
    )

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
