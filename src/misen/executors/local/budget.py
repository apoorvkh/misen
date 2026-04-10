"""Data models for the local executor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import msgspec

if TYPE_CHECKING:
    from misen.task_metadata import GpuRuntime, Resources


class ResourceBudget(msgspec.Struct, frozen=True):
    """Available resource budget for local scheduling.

    Values are interpreted as hard scheduler limits for concurrent job
    placement.
    """

    memory: int
    cpus: int
    cuda_gpus: int
    rocm_gpus: int
    xpu_gpus: int

    def fits(self, resources: Resources) -> bool:
        """Return whether requested resources fit in current budget."""
        return (
            resources.cpus <= self.cpus
            and resources.memory <= self.memory
            and resources.gpus <= self._runtime_gpu_budget(resources.gpu_runtime)
        )

    def subtract(self, resources: Resources) -> ResourceBudget:
        """Return new budget after reserving resources."""
        return self._adjust(resources=resources, multiplier=-1)

    def add(self, resources: Resources) -> ResourceBudget:
        """Return new budget after releasing resources."""
        return self._adjust(resources=resources, multiplier=1)

    def _adjust(self, resources: Resources, multiplier: Literal[-1, 1]) -> ResourceBudget:
        runtime_delta = resources.gpus * multiplier
        return ResourceBudget(
            cpus=self.cpus + (resources.cpus * multiplier),
            memory=self.memory + (resources.memory * multiplier),
            cuda_gpus=self.cuda_gpus + (runtime_delta if resources.gpu_runtime == "cuda" else 0),
            rocm_gpus=self.rocm_gpus + (runtime_delta if resources.gpu_runtime == "rocm" else 0),
            xpu_gpus=self.xpu_gpus + (runtime_delta if resources.gpu_runtime == "xpu" else 0),
        )

    def _runtime_gpu_budget(self, gpu_runtime: GpuRuntime) -> int:
        match gpu_runtime:
            case "cuda":
                return self.cuda_gpus
            case "rocm":
                return self.rocm_gpus
            case "xpu":
                return self.xpu_gpus
