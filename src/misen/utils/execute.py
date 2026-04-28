"""CLI entrypoint for executing serialized work-unit payloads.

The worker process runs this module to execute a single work unit.  It also
owns the **job-log lifecycle**: the parent (executor) tells the worker where to
write its log via ``--job-log-path``; the worker wraps its full lifecycle in
``workspace.streaming_job_log(...)`` so a remote workspace can publish the
local file to shared storage as it grows.  This way the same code captures
every byte the worker emits -- allocation/setup, ``WorkUnit.execute``,
post-execute finalizers -- regardless of where the worker is running.
"""

import base64
import binascii
import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

import cloudpickle
import tyro

from misen.task_metadata import GpuRuntime
from misen.utils.assigned_resources import AssignedResources, AssignedResourcesPerNode
from misen.utils.resource_binding import apply_resource_binding

if TYPE_CHECKING:
    from collections.abc import Callable


def execute(
    payload: Path,
    assigned_resources_getter: str,
    gpu_runtime: GpuRuntime = "cuda",
    *,
    bind_gpu_env: bool = True,
    job_log_path: Path | None = None,
) -> None:
    """Execute a cloudpickle work-unit payload file.

    Args:
        payload: Path to payload bytes; the payload is a dict with keys
            ``workspace`` (Workspace) and ``fn`` (callable accepting
            ``assigned_resources=``). The workspace is exposed so the
            worker can wrap its lifecycle in
            :meth:`Workspace.streaming_job_log` for live log publishing.
        assigned_resources_getter: URL-safe base64 string containing a
            cloudpickled assigned-resources getter callable.
        gpu_runtime: Expected runtime for GPU resources (if any). Defaults
            to ``"cuda"`` for backward compatibility with older payload runners.
        bind_gpu_env: Whether to apply GPU visibility environment variables
            from assigned resources. Schedulers such as SLURM may already set
            these correctly for the job step, so backends can disable this.
        job_log_path: Path where the parent executor is writing this worker's
            combined stdout/stderr log. When provided, the workspace can stream
            the log while the worker is still running.
    """
    assigned_resources: AssignedResources | AssignedResourcesPerNode | None = _get_assigned_resources(
        encoded_getter=assigned_resources_getter
    )
    apply_resource_binding(
        assigned_resources=assigned_resources,
        gpu_runtime=gpu_runtime,
        bind_gpu_env=bind_gpu_env,
    )

    bundle = cloudpickle.loads(payload.read_bytes())
    workspace = bundle["workspace"]
    payload_fn: Callable[..., None] = bundle["fn"]

    # The parent points the scheduler's stdout (``Popen(stdout=...)`` /
    # SLURM ``--output=...``) at this same path, so the live uploader
    # sees everything the worker writes -- allocation/setup,
    # ``WorkUnit.execute``, post-execute Python finalizers.
    streaming = workspace.streaming_job_log(job_log_path) if job_log_path is not None else contextlib.nullcontext()

    with streaming:
        payload_fn(assigned_resources=assigned_resources)


def _get_assigned_resources(encoded_getter: str) -> AssignedResources | AssignedResourcesPerNode | None:
    """Decode and unpickle an assigned-resources getter callable."""
    try:
        payload = base64.urlsafe_b64decode(encoded_getter.encode("ascii"))
    except (UnicodeEncodeError, binascii.Error) as exc:
        msg = "Invalid --assigned-resources-getter payload: expected URL-safe base64."
        raise ValueError(msg) from exc

    getter: Callable[[], AssignedResources | AssignedResourcesPerNode | None] = cloudpickle.loads(payload)
    if not callable(getter):
        msg = "Invalid --assigned-resources-getter payload: decoded object is not callable."
        raise TypeError(msg)
    assigned_resources: AssignedResources | AssignedResourcesPerNode | None = getter()
    return assigned_resources


if __name__ == "__main__":
    tyro.cli(execute)
