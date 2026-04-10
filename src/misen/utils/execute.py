"""CLI entrypoint for executing serialized work-unit payloads."""

import base64
import binascii
from pathlib import Path
from typing import TYPE_CHECKING

import cloudpickle
import tyro

from misen.task_metadata import GpuRuntime
from misen.utils.assigned_resources import AssignedResources, AssignedResourcesPerNode
from misen.utils.resource_binding import apply_resource_binding

if TYPE_CHECKING:
    from collections.abc import Callable


def execute(payload: Path, assigned_resources_getter: str, gpu_runtime: GpuRuntime = "cuda") -> None:
    """Execute a cloudpickle payload file.

    Args:
        payload: Path to payload bytes representing a zero-argument callable.
        assigned_resources_getter: URL-safe base64 string containing a
            cloudpickled assigned-resources getter callable.
        gpu_runtime: Expected runtime for GPU resources (if any). Defaults
            to ``"cuda"`` for backward compatibility with older payload runners.
    """
    assigned_resources: AssignedResources | AssignedResourcesPerNode | None = _get_assigned_resources(
        encoded_getter=assigned_resources_getter
    )
    apply_resource_binding(assigned_resources=assigned_resources, gpu_runtime=gpu_runtime)

    payload_fn: Callable[[AssignedResources | AssignedResourcesPerNode | None], None]
    payload_fn = cloudpickle.loads(payload.read_bytes())
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
