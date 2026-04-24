"""Tests for ``Task.with_resources`` override helper."""

import pytest

from misen import ASSIGNED_RESOURCES_PER_NODE, Resources, Task, meta
from misen.utils.work_unit import build_work_graph


@meta(id="noop_task", cache=False, resources=Resources(gpus=1, gpu_memory=16))
def noop_task(x: int) -> int:
    return x


@meta(id="multinode_task", cache=False, exclude={"x"}, resources={"nodes": 2, "gpus": 1})
def multinode_task(x: object) -> None:
    _ = x


@meta(id="resource_override_dependency", cache=True, resources={"cpus": 1})
def resource_override_dependency() -> int:
    return 1


@meta(id="resource_override_left", cache=False)
def resource_override_left(x: int) -> int:
    return x


@meta(id="resource_override_right", cache=False)
def resource_override_right(x: int) -> int:
    return x


@meta(id="resource_override_pair", cache=False)
def resource_override_pair(values: tuple[int, int]) -> int:
    return sum(values)


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


def test_dependency_collection_merges_resource_overrides_for_equal_tasks() -> None:
    base = Task(resource_override_dependency)
    patched = base.with_resources(cpus=8)

    task = Task(resource_override_pair, values=(base.T, patched.T))

    assert len(task.dependencies) == 1
    (dependency,) = task.dependencies
    assert dependency.resources["cpus"] == 8


def test_work_graph_merges_resource_overrides_for_equal_dependencies() -> None:
    base = Task(resource_override_dependency)
    patched = base.with_resources(cpus=8)

    work_graph = build_work_graph(
        {
            Task(resource_override_left, x=base.T),
            Task(resource_override_right, x=patched.T),
        }
    )

    dependency_units = [work_unit for work_unit in work_graph.nodes() if work_unit.root == base]
    assert len(dependency_units) == 1
    assert dependency_units[0].resources["cpus"] == 8
