"""Cooperative resource controls supplied in worker launch environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from misen.task_metadata import AcceleratorType

__all__ = ["resource_environment"]

_ACCELERATOR_MASK_VARS: dict[AcceleratorType, tuple[str, ...]] = {
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


def resource_environment(
    cpu_indices: list[int] | None = None,
    accelerator_type: AcceleratorType = "cuda",
    accelerator_indices: list[int] | None = None,
) -> dict[str, str]:
    """Return launch variables for an executor's assigned resources.

    ``None`` preserves scheduler-managed thread counts or device visibility;
    an empty accelerator list hides all maskable devices. These variables are
    cooperative runtime controls rather than a device-access security boundary;
    later environment activation or task code can replace them.
    """
    env = dict(_DYNAMIC_THREAD_DISABLE_ENV)
    if cpu_indices is not None:
        env.update(dict.fromkeys(_CPU_THREAD_CAP_VARS, str(len(cpu_indices))))
    if accelerator_indices is not None:
        mask = ",".join(map(str, accelerator_indices))
        env.update(dict.fromkeys(_ACCELERATOR_MASK_VARS.get(accelerator_type, ()), mask))
    return env
