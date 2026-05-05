"""Runtime resource binding applied before user payload initialization."""

from __future__ import annotations

import os
from contextlib import suppress
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from collections.abc import Mapping

    from misen.task_metadata import GpuRuntime

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
    cpu_indices: list[int] | None,
    gpu_indices: list[int] | None,
    gpu_runtime: GpuRuntime,
) -> None:
    """Bind current process to assigned resources.

    Each resource type is bound only when the executor passes explicit indices.
    Backends that delegate isolation to a scheduler (e.g. ``SlurmExecutor``,
    where SLURM cgroups already mask GPUs and pin CPU affinity) pass ``None``
    so the worker leaves the inherited environment untouched. The runtime view
    (``CUDA_VISIBLE_DEVICES``, ``os.sched_getaffinity``) is the standardized
    interface user code reads to discover its allotment.
    """
    for key, value in _DYNAMIC_THREAD_DISABLE_ENV.items():
        os.environ[key] = value

    if cpu_indices is not None:
        cpu_count_str = str(len(cpu_indices))
        for key in _CPU_THREAD_CAP_VARS:
            os.environ[key] = cpu_count_str
        _apply_cpu_affinity(cpu_indices)

    if gpu_indices is not None:
        gpu_mask = ",".join(str(index) for index in gpu_indices)
        for key in _GPU_MASK_VARS[gpu_runtime]:
            os.environ[key] = gpu_mask


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
