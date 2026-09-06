"""Declared SkyPilot capacity validation and resource fitting."""
# ruff: noqa: D103, PLR2004, S101

from __future__ import annotations

import tomllib
from typing import Any, cast

import msgspec
import pytest

from misen.executors.skypilot import SkyPilotCapacity
from misen.task_metadata import Resources


def test_toml_capacity_profiles_decode() -> None:
    config = tomllib.loads(
        """
        [capacity.cpu]
        pool = "misen-cpu"
        cpus = 8
        memory = 32
        max_workers = 3

        [capacity.gpu]
        infra = ["aws/us-east-1", "aws/us-west-2"]
        cpus = 8
        memory = 64
        accelerators = {A100 = 1}
        accelerator_memory = 80
        use_spot = true
        """
    )
    profiles = msgspec.convert(config["capacity"], type=dict[str, SkyPilotCapacity])

    assert profiles["cpu"].pool == "misen-cpu"
    assert profiles["cpu"].max_workers == 3
    assert profiles["cpu"].borrowed
    assert profiles["gpu"].accelerators == {"A100": 1}
    assert profiles["gpu"].accelerator_memory == 80
    assert not profiles["gpu"].borrowed


def test_defaults_declare_a_small_cpu_reservation() -> None:
    capacity = SkyPilotCapacity(infra="aws")

    assert capacity.cpus == 1
    assert capacity.memory == 8
    assert capacity.max_workers == 1
    assert capacity.nodes == 1
    assert capacity.accelerator_count == 0
    assert capacity.fits(Resources())
    assert not capacity.fits(Resources(cpus=2))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"pool": "workers", "cluster": "workers"},
        {"pool": "workers", "infra": "aws"},
        {"cluster": "workers", "infra": "aws"},
        {"pool": "workers", "cluster": "workers", "infra": "aws"},
    ],
)
def test_exactly_one_source_is_required(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Exactly one capacity source"):
        SkyPilotCapacity(**kwargs)


@pytest.mark.parametrize("source", ["pool", "cluster"])
@pytest.mark.parametrize("value", ["", "bad/name", "trailing-", "with space", "x" * 64, True, 123])
def test_source_names_are_validated(source: str, value: object) -> None:
    with pytest.raises(ValueError, match=source):
        SkyPilotCapacity(**{source: value})


def test_pool_and_cluster_name_formats_remain_distinct() -> None:
    assert SkyPilotCapacity(pool="Research_Pool.2").pool == "Research_Pool.2"
    with pytest.raises(ValueError, match="cluster"):
        SkyPilotCapacity(cluster="Research_Pool.2")


@pytest.mark.parametrize("value", ["", " ", [], ["aws", " aws "], [None], False, ("aws",), "aws\nsecret"])
def test_infrastructure_alternatives_are_validated(value: object) -> None:
    with pytest.raises(ValueError, match="infra"):
        SkyPilotCapacity(infra=cast("Any", value))


def test_infrastructure_and_option_strings_are_normalized() -> None:
    capacity = SkyPilotCapacity(
        infra=[" aws/us-east-1 ", " azure/eastus "],
        accelerators={" A100 ": 1},
        instance_type=" p4d.24xlarge ",
        image_id=" ami-example ",
    )

    assert capacity.infra == ["aws/us-east-1", "azure/eastus"]
    assert capacity.accelerators == {"A100": 1}
    assert capacity.instance_type == "p4d.24xlarge"
    assert capacity.image_id == "ami-example"


@pytest.mark.parametrize("name", ["cpus", "memory", "max_workers", "nodes", "disk_size", "accelerator_memory"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, float("inf"), float("nan"), "1"])
def test_capacity_limits_require_positive_integers(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=name):
        SkyPilotCapacity(infra="aws", **{name: value})


@pytest.mark.parametrize("name", ["dedicated", "use_spot"])
@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_boolean_options_reject_nonbooleans(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=name):
        SkyPilotCapacity(infra="aws", **{name: value})


@pytest.mark.parametrize("value", [{"A100": 1, "L4": 1}, ["A100"], None])
def test_at_most_one_concrete_accelerator_model(value: object) -> None:
    with pytest.raises(ValueError, match="accelerators"):
        SkyPilotCapacity(infra="aws", accelerators=cast("Any", value))


@pytest.mark.parametrize("model", ["", "A100:1", "A 100", "A100\nX", 123])
def test_invalid_accelerator_model(model: object) -> None:
    with pytest.raises(ValueError, match="accelerator model"):
        SkyPilotCapacity(infra="aws", accelerators=cast("Any", {model: 1}))


@pytest.mark.parametrize("count", [0, -1, True, 1.5, float("inf")])
def test_accelerator_count_is_positive(count: object) -> None:
    with pytest.raises(ValueError, match="accelerator count"):
        SkyPilotCapacity(infra="aws", accelerators=cast("Any", {"A100": count}))


def test_accelerator_memory_requires_devices() -> None:
    with pytest.raises(ValueError, match="requires an accelerator model"):
        SkyPilotCapacity(infra="aws", accelerator_memory=80)


def test_unsupported_accelerator_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported accelerator type"):
        SkyPilotCapacity(infra="aws", accelerator_type=cast("Any", "gpu"))


@pytest.mark.parametrize("source", [{"pool": "workers"}, {"cluster": "workers"}])
@pytest.mark.parametrize(
    "option", [{"use_spot": True}, {"image_id": "ami-1"}, {"instance_type": "m6i.large"}, {"disk_size": 100}]
)
def test_borrowed_sources_reject_creation_options(source: dict[str, Any], option: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Borrowed pool/cluster capacity"):
        SkyPilotCapacity(**source, **option)


@pytest.mark.parametrize("name", ["instance_type", "image_id"])
@pytest.mark.parametrize("value", ["", " ", 123])
def test_creation_string_options_are_validated(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=name):
        SkyPilotCapacity(infra="aws", **{name: value})


def test_single_borrowed_cluster_is_one_reservation() -> None:
    with pytest.raises(ValueError, match="max_workers=1"):
        SkyPilotCapacity(cluster="workers", max_workers=2)


def test_multinode_capacity_requires_dedicated_reservation() -> None:
    with pytest.raises(ValueError, match="dedicated=True"):
        SkyPilotCapacity(infra="aws", nodes=2)
    capacity = SkyPilotCapacity(infra="aws", nodes=2, dedicated=True, cpus=4, memory=16)
    assert capacity.fits(Resources(nodes=2, cpus=4, memory=16))
    assert capacity.fits(Resources(nodes=1, cpus=1, memory=1))
    assert not capacity.fits(Resources(nodes=3))


@pytest.mark.parametrize(
    ("resources", "fits"),
    [
        (Resources(cpus=8, memory=32), True),
        (Resources(cpus=9), False),
        (Resources(memory=33), False),
        (Resources(nodes=2), False),
        (Resources(accelerators=1), False),
        (Resources(time=100_000), True),
    ],
)
def test_cpu_resource_fitting(resources: Resources, *, fits: bool) -> None:
    assert SkyPilotCapacity(pool="cpu", cpus=8, memory=32).fits(resources) is fits


@pytest.mark.parametrize(
    ("resources", "fits"),
    [
        (Resources(), True),
        (Resources(accelerators=2), True),
        (Resources(accelerators=3), False),
        (Resources(accelerators=1, accelerator_memory=80), True),
        (Resources(accelerators=1, accelerator_memory=81), False),
        (Resources(accelerators=1, accelerator_type="rocm"), False),
        (Resources(accelerators=0, accelerator_type="rocm"), True),
    ],
)
def test_accelerator_resource_fitting(resources: Resources, *, fits: bool) -> None:
    capacity = SkyPilotCapacity(pool="gpu", cpus=8, memory=32, accelerators={"A100": 2}, accelerator_memory=80)
    assert capacity.accelerator_count == 2
    assert capacity.fits(resources) is fits


def test_unknown_device_memory_cannot_satisfy_explicit_minimum() -> None:
    capacity = SkyPilotCapacity(infra="aws", accelerators={"A100": 1})
    assert capacity.fits(Resources(accelerators=1))
    assert not capacity.fits(Resources(accelerators=1, accelerator_memory=40))


@pytest.mark.parametrize("resources", [Resources(cpus=True), Resources(accelerators=-1), Resources(memory=0)])
def test_invalid_requests_are_not_silently_admitted(resources: Resources) -> None:
    with pytest.raises(ValueError, match=r"must be a (positive|nonnegative) integer"):
        SkyPilotCapacity(infra="aws").fits(resources)


def test_owned_options_declare_shape_and_creation_parameters() -> None:
    capacity = SkyPilotCapacity(
        infra="aws/us-east-1",
        cpus=8,
        memory=64,
        accelerators={"A100": 1},
        accelerator_memory=80,
        use_spot=True,
        instance_type="p4d.24xlarge",
        image_id="ami-1",
        disk_size=100,
    )
    assert capacity.as_sky_options() == {
        "infra": "aws/us-east-1",
        "cpus": "8+",
        "memory": "64+",
        "accelerators": {"A100": 1},
        "use_spot": True,
        "instance_type": "p4d.24xlarge",
        "image_id": "ami-1",
        "disk_size": 100,
    }


def test_borrowed_options_exclude_creation_parameters() -> None:
    capacity = SkyPilotCapacity(pool="cpu", cpus=4, memory=16)
    assert capacity.as_sky_options() == {"cpus": "4+", "memory": "16+"}


def test_options_and_accelerator_defaults_do_not_share_mutable_state() -> None:
    first = SkyPilotCapacity(infra="aws")
    second = SkyPilotCapacity(infra="aws")
    assert first.accelerators is not second.accelerators
    capacity = SkyPilotCapacity(infra=["aws", "azure"], accelerators={"A100": 1})
    options = capacity.as_sky_options()
    options["accelerators"]["A100"] = 2
    options["infra"].append("gcp")
    assert capacity.accelerators == {"A100": 1}
    assert capacity.infra == ["aws", "azure"]


def test_msgspec_config_rejects_unknown_fields_and_invalid_limits() -> None:
    with pytest.raises(msgspec.ValidationError, match="unknown field"):
        msgspec.convert({"infra": "aws", "max_worker": 1}, type=SkyPilotCapacity)
    with pytest.raises(msgspec.ValidationError, match="positive integer"):
        msgspec.convert({"infra": "aws", "max_workers": 0}, type=SkyPilotCapacity)
    with pytest.raises(msgspec.ValidationError, match="Expected `int`, got `bool`"):
        msgspec.convert({"infra": "aws", "cpus": True}, type=SkyPilotCapacity)
