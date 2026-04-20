"""Tests for ``Task.with_resources`` override helper."""

import pytest

from misen import ASSIGNED_RESOURCES_PER_NODE, Resources, Task, meta


@meta(id="noop_task", cache=False, resources=Resources(gpus=1, gpu_memory=16))
def noop_task(x: int) -> int:
    return x


@meta(id="multinode_task", cache=False, exclude={"x"}, resources={"nodes": 2, "gpus": 1})
def multinode_task(x: object) -> None:
    _ = x


def test_with_resources_overrides_only_named_fields() -> None:
    task = Task(noop_task, x=1)
    patched = task.with_resources(gpu_memory=40)

    assert patched.resources["gpu_memory"] == 40
    assert patched.resources["gpus"] == task.resources["gpus"]
    assert patched.resources["memory"] == task.resources["memory"]
    assert patched.resources["cpus"] == task.resources["cpus"]


def test_with_resources_preserves_task_identity() -> None:
    task = Task(noop_task, x=1)
    patched = task.with_resources(gpu_memory=40)

    assert patched.task_hash() == task.task_hash()
    assert patched.meta is task.meta
    assert patched.args == task.args
    assert patched.kwargs == task.kwargs
    assert patched.dependencies == task.dependencies


def test_with_resources_returns_new_instance_and_does_not_mutate() -> None:
    task = Task(noop_task, x=1)
    original_gpu_memory = task.resources["gpu_memory"]
    patched = task.with_resources(gpu_memory=40)

    assert patched is not task
    assert task.resources["gpu_memory"] == original_gpu_memory


def test_with_resources_rejects_unknown_field() -> None:
    task = Task(noop_task, x=1)
    with pytest.raises(TypeError):
        task.with_resources(not_a_field=1)  # ty:ignore[unknown-argument]


def test_with_resources_rejects_per_node_sentinel_when_nodes_collapse_to_one() -> None:
    task = Task(multinode_task, x=ASSIGNED_RESOURCES_PER_NODE)
    with pytest.raises(ValueError, match=r"ASSIGNED_RESOURCES_PER_NODE"):
        task.with_resources(nodes=1)
