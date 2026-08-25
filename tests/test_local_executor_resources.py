"""Local accelerator request resolution."""
# ruff: noqa: D103, PLR2004, S101

from pathlib import Path
from typing import TYPE_CHECKING, cast

import cloudpickle
import pytest

import misen.executors.local as local_module
from misen import Resources, SubmissionError, Task, meta
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


def test_local_executor_uses_the_inherited_cpu_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_module.os, "sched_getaffinity", lambda _pid: {3, 7})

    executor = LocalExecutor(num_cpus=1)

    assert executor._scheduler.available_cpu_indices == [3]  # noqa: SLF001


def test_local_executor_rejects_cpus_outside_the_inherited_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_module.os, "sched_getaffinity", lambda _pid: {3, 7})

    with pytest.raises(ValueError, match="unavailable"):
        LocalExecutor(cpu_indices=[1])


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
    with pytest.raises(SubmissionError, match="cannot verify"):
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

    with pytest.raises(SubmissionError, match="only single-node"):
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


def test_local_executor_builds_launch_environment_from_its_pool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executor = LocalExecutor(accelerators=1, accelerator_type="rocm")
    work_unit = WorkUnit(root=Task(_local_resource_test), dependencies=set())
    job = LocalJob(work_unit, dependencies=set(), snapshot=None, workspace=cast("Workspace", None))
    prepare_kwargs: dict[str, object] = {}
    resource_args: tuple[object, ...] | None = None

    def capture_prepare(**kwargs: object) -> tuple[str, list[str], dict[str, str], Path]:
        prepare_kwargs.update(kwargs)
        return "job-id", ["python"], {}, tmp_path / "job.log"

    def capture_resource_environment(*args: object) -> dict[str, str]:
        nonlocal resource_args
        resource_args = args
        msg = "captured"
        raise RuntimeError(msg)

    monkeypatch.setattr(local_module, "prepare_live_job", capture_prepare)
    monkeypatch.setattr(local_module, "resource_environment", capture_resource_environment)
    with pytest.raises(RuntimeError, match="captured"):
        executor._scheduler._launch_job(job, cpu_indices=[0], accelerator_indices=[])  # noqa: SLF001

    assert set(prepare_kwargs) == {"work_unit", "workspace"}
    assert resource_args == ([0], "rocm", [])
