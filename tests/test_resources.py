"""Device resource defaults and aggregation semantics."""
# ruff: noqa: D103, PLR2004, S101

from typing import Any, cast

import pytest

from misen import AcceleratorType, Resources, Task, meta
from misen.task_metadata import aggregate_resources


def _resources_for_parallelism(parallelism: int) -> Resources:
    return Resources(accelerators=parallelism, accelerator_memory=40)


def _resources_for_type(accelerator_type: AcceleratorType) -> Resources:
    return Resources(accelerators=1, accelerator_type=accelerator_type)


@meta(id="argument_specific_resources", resources=_resources_for_parallelism)
def _argument_specific_resources(parallelism: int) -> int:
    return parallelism


@meta(id="argument_specific_accelerator_type", resources=_resources_for_type)
def _argument_specific_accelerator_type(accelerator_type: AcceleratorType) -> AcceleratorType:
    return accelerator_type


@meta(id="default_resources")
def _default_resources() -> None:
    return None


def _resolved(**overrides: Any) -> Resources:
    return cast("Resources", {**Task(_default_resources).resources, **overrides})


def test_device_defaults() -> None:
    resources = Task(_default_resources).resources

    assert resources["accelerators"] == 0
    assert resources["accelerator_memory"] is None
    assert resources["accelerator_type"] == "cuda"


def test_device_request_can_be_computed_from_task_arguments() -> None:
    task = Task(_argument_specific_resources, parallelism=4)

    assert task.resources["accelerators"] == 4
    assert task.resources["accelerator_memory"] == 40


def test_accelerator_type_can_be_computed_from_task_arguments() -> None:
    task = Task(_argument_specific_accelerator_type, accelerator_type="tpu")

    assert task.resources["accelerator_type"] == "tpu"


def test_device_requirements_are_merged() -> None:
    first = _resolved(time=10, accelerators=4, accelerator_memory=40)
    second = _resolved(time=20, accelerators=2, accelerator_memory=80)

    aggregate = aggregate_resources([first, second])

    assert aggregate["time"] == 30
    assert aggregate["accelerators"] == 4
    assert aggregate["accelerator_memory"] == 80

    alternatives = aggregate_resources([first, second], sum_time=False)
    assert alternatives["time"] == 20


def test_aggregation_rejects_different_accelerator_types() -> None:
    with pytest.raises(ValueError, match="Incompatible accelerator types"):
        aggregate_resources(
            [_resolved(accelerators=1, accelerator_type="cuda"), _resolved(accelerators=1, accelerator_type="tpu")]
        )


def test_accelerator_memory_requires_a_device() -> None:
    with pytest.raises(ValueError, match="requires accelerators > 0"):
        Task(_default_resources).with_resources(accelerator_memory=40)
