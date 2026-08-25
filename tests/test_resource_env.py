"""Launch-time resource environment behavior."""
# ruff: noqa: ANN001, D103, S101

import os

import pytest

from misen.utils.resource_env import resource_environment

_THREAD_CAP_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TBB_NUM_THREADS",
)


def test_resource_environment_sets_thread_caps_for_explicit_cpus() -> None:
    env = resource_environment(cpu_indices=[0, 1, 2])

    assert all(env[name] == "3" for name in _THREAD_CAP_VARS)
    assert env["OMP_DYNAMIC"] == "FALSE"
    assert env["MKL_DYNAMIC"] == "FALSE"
    assert env["OPENBLAS_DYNAMIC"] == "0"


def test_resource_environment_preserves_scheduler_managed_thread_counts() -> None:
    env = resource_environment(cpu_indices=None)

    assert not set(_THREAD_CAP_VARS).intersection(env)
    assert env["OMP_DYNAMIC"] == "FALSE"


@pytest.mark.parametrize(
    ("accelerator_type", "variables"),
    [
        ("cuda", ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")),
        ("rocm", ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")),
        ("xpu", ("ZE_AFFINITY_MASK",)),
    ],
)
def test_resource_environment_masks_assigned_accelerators(accelerator_type, variables) -> None:
    env = resource_environment(accelerator_type=accelerator_type, accelerator_indices=[3])

    assert all(env[variable] == "3" for variable in variables)


def test_resource_environment_distinguishes_empty_from_scheduler_managed() -> None:
    inherited = resource_environment(accelerator_type="cuda", accelerator_indices=None)
    hidden = resource_environment(accelerator_type="cuda", accelerator_indices=[])

    assert "CUDA_VISIBLE_DEVICES" not in inherited
    assert hidden["CUDA_VISIBLE_DEVICES"] == ""
    assert hidden["NVIDIA_VISIBLE_DEVICES"] == ""


def test_resource_environment_does_not_mutate_the_process_environment(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "inherited")
    before = os.environ.copy()

    resource_environment(cpu_indices=[0])

    assert os.environ == before
