"""Cooperative resource controls supplied in worker launch environments."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from misen.task_metadata import AcceleratorType

__all__ = ["narrow_accelerator_environment", "resource_environment"]

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


_DEVICE_MASKS = {
    "cuda": ("CUDA_VISIBLE_DEVICES",),
    "rocm": ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"),
    "xpu": ("ZE_AFFINITY_MASK",),
}


def narrow_accelerator_environment(base: dict[str, str], overrides: dict[str, str]) -> dict[str, str]:
    """Narrow scheduler-assigned device masks without inventing physical IDs."""
    env: dict[str, str] = {**base, **overrides}
    count_value = overrides.get("MISEN_ACCELERATOR_COUNT")
    if count_value is None:
        return env
    env.pop("MISEN_ACCELERATOR_COUNT", None)
    env.pop("MISEN_ACCELERATOR_TYPE", None)
    if not re.fullmatch(r"[0-9]{1,6}", count_value):
        msg = "MISEN_ACCELERATOR_COUNT must be a bounded nonnegative integer."
        raise ValueError(msg)
    count = int(count_value)
    if count == 0:
        for masks in _DEVICE_MASKS.values():
            for name in masks:
                env[name] = ""
        return env
    masks = _DEVICE_MASKS.get(overrides.get("MISEN_ACCELERATOR_TYPE", ""))
    if masks is None:
        msg = "This accelerator backend has no supported worker device-mask isolation."
        raise ValueError(msg)
    assigned = False
    for name in masks:
        env.pop(name, None)  # A task override must never widen the scheduler's reservation.
        if name not in base:
            continue
        devices = base[name].split(",")
        if (
            len(devices) < count
            or len(set(devices)) != len(devices)
            or any(
                not device or device.strip() != device or device.lower() in {"-1", "all", "none", "void"}
                for device in devices
            )
        ):
            msg = "The inherited scheduler device mask cannot satisfy this accelerator request."
            raise ValueError(msg)
        env[name] = ",".join(devices[:count])
        assigned = True
    if not assigned:
        msg = "Accelerator work requires an inherited scheduler device mask; no device IDs will be guessed."
        raise ValueError(msg)
    return env
