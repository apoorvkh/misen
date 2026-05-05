import os

from misen.utils.resource_binding import apply_resource_binding


def test_apply_resource_binding_masks_gpu_env_when_indices_given(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "0")

    apply_resource_binding(cpu_indices=None, gpu_indices=[3], gpu_runtime="cuda")

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "3"


def test_apply_resource_binding_leaves_gpu_env_alone_when_indices_none(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "0")

    apply_resource_binding(cpu_indices=None, gpu_indices=None, gpu_runtime="cuda")

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "0"


def test_apply_resource_binding_sets_thread_caps_when_cpu_indices_given(monkeypatch) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    apply_resource_binding(cpu_indices=[0, 1, 2], gpu_indices=None, gpu_runtime="cuda")

    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["MKL_NUM_THREADS"] == "3"


def test_apply_resource_binding_leaves_thread_caps_alone_when_cpu_indices_none(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "42")

    apply_resource_binding(cpu_indices=None, gpu_indices=None, gpu_runtime="cuda")

    # Thread caps untouched (only the dynamic-disable env vars are set
    # unconditionally as a safety baseline).
    assert os.environ["OMP_NUM_THREADS"] == "42"
    assert os.environ["OMP_DYNAMIC"] == "FALSE"
