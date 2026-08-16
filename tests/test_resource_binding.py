"""Runtime resource-binding behavior."""
# ruff: noqa: ANN001, D103, S101

import os

import pytest

from misen.utils.resource_binding import apply_resource_binding


def test_apply_resource_binding_sets_thread_caps_when_cpu_indices_given(monkeypatch) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    apply_resource_binding(cpu_indices=[0, 1, 2])

    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["MKL_NUM_THREADS"] == "3"


def test_apply_resource_binding_leaves_thread_caps_alone_when_cpu_indices_none(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "42")

    apply_resource_binding(cpu_indices=None)

    # Thread caps untouched (only the dynamic-disable env vars are set
    # unconditionally as a safety baseline).
    assert os.environ["OMP_NUM_THREADS"] == "42"
    assert os.environ["OMP_DYNAMIC"] == "FALSE"


@pytest.mark.parametrize(
    ("accelerator_type", "variables"),
    [
        ("cuda", ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")),
        ("rocm", ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")),
        ("xpu", ("ZE_AFFINITY_MASK",)),
    ],
)
def test_apply_resource_binding_masks_assigned_accelerators(monkeypatch, accelerator_type, variables) -> None:
    for variable in variables:
        monkeypatch.setenv(variable, "inherited")

    apply_resource_binding(cpu_indices=None, accelerator_type=accelerator_type, accelerator_indices=[3])

    assert all(os.environ[variable] == "3" for variable in variables)


def test_apply_resource_binding_distinguishes_empty_from_scheduler_managed(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "inherited")

    apply_resource_binding(cpu_indices=None, accelerator_type="cuda", accelerator_indices=None)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "inherited"

    apply_resource_binding(cpu_indices=None, accelerator_type="cuda", accelerator_indices=[])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
