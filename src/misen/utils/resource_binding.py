"""Runtime resource binding applied before user payload initialization."""

from __future__ import annotations

import os
from contextlib import suppress
from typing import TYPE_CHECKING

import psutil

from misen.utils.assigned_resources import (
    AssignedResources,
    AssignedResourcesPerNode,
    select_local_assigned_resources,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from misen.task_properties import GpuRuntime

__all__ = ["apply_resource_binding"]

_GPU_MASK_VARS: Mapping[GpuRuntime, tuple[str, ...]] = {
    "cuda": ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"),
    "rocm": ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"),
    "xpu": ("ZE_AFFINITY_MASK",),
}

_CPU_THREAD_CAP_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TBB_NUM_THREADS",
)

_DYNAMIC_THREAD_DISABLE_ENV = {
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "OPENBLAS_DYNAMIC": "0",
}


def apply_resource_binding(
    assigned_resources: AssignedResources | AssignedResourcesPerNode | None, gpu_runtime: GpuRuntime
) -> None:
    """Bind current process to assigned resources (CPU/GPU/threading)."""
    local_resources = select_local_assigned_resources(assigned_resources)
    env_overrides = _binding_env_overrides(local_resources=local_resources, gpu_runtime=gpu_runtime)

    for key, value in env_overrides.items():
        os.environ[key] = value

    if local_resources is not None:
        _apply_cpu_affinity(local_resources["cpu_indices"])


def _binding_env_overrides(local_resources: AssignedResources | None, gpu_runtime: GpuRuntime) -> dict[str, str]:
    """Compute process-environment overrides for assigned resources."""
    env = dict(_DYNAMIC_THREAD_DISABLE_ENV)
    if local_resources is None:
        return env

    cpu_count = len(local_resources["cpu_indices"])
    if cpu_count > 0:
        cpu_count_str = str(cpu_count)
        for key in _CPU_THREAD_CAP_VARS:
            env[key] = cpu_count_str

    gpu_mask = ",".join(str(index) for index in local_resources["gpu_indices"])
    for key in _GPU_MASK_VARS[gpu_runtime]:
        env[key] = gpu_mask

    return env


def _apply_cpu_affinity(cpu_indices: list[int]) -> None:
    """Set process CPU affinity when the platform supports it."""
    if not cpu_indices:
        return

    sched_setaffinity = getattr(os, "sched_setaffinity", None)
    if sched_setaffinity is not None:
        with suppress(AttributeError, ImportError, OSError, RuntimeError, ValueError):
            sched_setaffinity(0, set(cpu_indices))
            return

    process = psutil.Process()
    cpu_affinity = getattr(process, "cpu_affinity", None)
    if cpu_affinity is not None:
        with suppress(AttributeError, ImportError, OSError, RuntimeError, ValueError):
            cpu_affinity(cpu_indices)
