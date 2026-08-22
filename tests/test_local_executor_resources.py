"""Local accelerator request resolution."""
# ruff: noqa: D103, PLR2004, S101

from typing import TYPE_CHECKING, cast

import cloudpickle
import pytest

import misen.executors.local as local_module
from misen import Resources, Task, meta
from misen.executors.local import LocalExecutor, LocalJob
from misen.utils.work_unit import WorkUnit

if TYPE_CHECKING:
    from misen.workspace import Workspace


@meta(id="local_resource_test")
def _local_resource_test() -> None:
    return None


def _resources(**overrides: object) -> Resources:
    return Task(_local_resource_test).with_resources(**overrides).resources


def test_local_executor_defaults_an_unspecified_accelerator_type_to_cuda() -> None:
    assert _resources(accelerators=1)["accelerator_type"] == "cuda"


def test_local_executor_accepts_mps_as_an_accelerator_type() -> None:
    executor = LocalExecutor(accelerators=1, accelerator_type="mps")
    assert executor._resource_budget.fits(_resources(accelerators=1, accelerator_type="mps"))  # noqa: SLF001


def test_local_executor_exposes_one_configured_accelerator_pool() -> None:
    executor = LocalExecutor(accelerators=1, accelerator_type="mps")

    assert executor._resource_budget.fits(_resources(accelerators=1, accelerator_type="mps"))  # noqa: SLF001
    assert not executor._resource_budget.fits(_resources(accelerators=1, accelerator_type="cuda"))  # noqa: SLF001


def test_local_executor_rejects_multiple_mps_devices() -> None:
    with pytest.raises(ValueError, match="single GPU"):
        LocalExecutor(accelerators=2, accelerator_type="mps")


def test_local_executor_rejects_unverifiable_memory_requirement() -> None:
    executor = LocalExecutor(accelerators=1)
    with pytest.raises(ValueError, match="cannot verify"):
        executor._dispatch(  # noqa: SLF001
            work_unit=WorkUnit(
                root=Task(_local_resource_test).with_resources(accelerators=1, accelerator_memory=40),
                dependencies=set(),
            ),
            dependencies=set(),
            workspace=cast("Workspace", None),
            snapshot=None,
        )


def test_local_executor_rejects_multiple_nodes() -> None:
    executor = LocalExecutor()
    resources = _resources(nodes=2)

    with pytest.raises(ValueError, match="only single-node"):
        executor._dispatch(  # noqa: SLF001
            work_unit=WorkUnit(
                root=Task(_local_resource_test).with_resources(**resources),
                dependencies=set(),
            ),
            dependencies=set(),
            workspace=cast("Workspace", None),
            snapshot=None,
        )


def test_local_executor_provides_one_exclusive_tpu_pool() -> None:
    executor = LocalExecutor(accelerators=4, accelerator_type="tpu")
    request = _resources(accelerators=4, accelerator_type="tpu")

    assert executor._resource_budget.fits(request)  # noqa: SLF001
    assert not executor._resource_budget.fits(_resources(accelerators=2, accelerator_type="tpu"))  # noqa: SLF001
    reserved = executor._resource_budget.subtract(request)  # noqa: SLF001
    assert not reserved.fits(request)
    assert reserved.add(request) == executor._resource_budget  # noqa: SLF001


def test_local_executor_rejects_tpu_indices() -> None:
    with pytest.raises(ValueError, match="indices for TPUs"):
        LocalExecutor(accelerator_type="tpu", accelerator_indices=[0])


def test_local_executor_round_trips_through_cloudpickle() -> None:
    executor = LocalExecutor(accelerators=2, accelerator_type="rocm")
    restored = cloudpickle.loads(cloudpickle.dumps(executor))

    assert restored.accelerators == 2
    assert restored.accelerator_type == "rocm"


def test_local_executor_forwards_its_pool_binding_to_the_final_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = LocalExecutor(accelerators=1, accelerator_type="rocm")
    work_unit = WorkUnit(root=Task(_local_resource_test), dependencies=set())
    job = LocalJob(work_unit, dependencies=set(), snapshot=None, workspace=cast("Workspace", None))
    captured: dict[str, object] = {}

    def capture_prepare(**kwargs: object) -> None:
        captured.update(kwargs)
        msg = "captured"
        raise RuntimeError(msg)

    monkeypatch.setattr(local_module, "prepare_live_job", capture_prepare)
    with pytest.raises(RuntimeError, match="captured"):
        executor._scheduler._launch_job(job, cpu_indices=[0], accelerator_indices=[])  # noqa: SLF001

    assert captured["accelerator_type"] == "rocm"
    assert captured["accelerator_indices"] == []
