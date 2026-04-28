import os

from misen.utils.assigned_resources import AssignedResources
from misen.utils.resource_binding import apply_resource_binding


def _assigned(*, gpu_indices: list[int]) -> AssignedResources:
    return AssignedResources(
        cpu_indices=[],
        gpu_indices=gpu_indices,
        memory=None,
        gpu_memory=None,
    )


def test_apply_resource_binding_sets_gpu_env_by_default(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "0")

    apply_resource_binding(assigned_resources=_assigned(gpu_indices=[3]), gpu_runtime="cuda")

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "3"


def test_apply_resource_binding_can_preserve_scheduler_gpu_env(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "0")

    apply_resource_binding(
        assigned_resources=_assigned(gpu_indices=[3]),
        gpu_runtime="cuda",
        bind_gpu_env=False,
    )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "0"
