"""Tests for task argument hashing and result indexing behavior."""

import inspect
from pathlib import Path
from typing import Any

import dill
import pytest

from misen import HashError, Task, meta
from misen.task_metadata import TaskMetadata
from misen.utils.hashing import ResultHash
from misen.utils.serde import Serializer
from misen.workspaces.disk import DiskWorkspace


class DillSerializer(Serializer[Any]):
    """Test-only serializer using dill for arbitrary types."""

    @staticmethod
    def save(obj: Any, directory: Path) -> None:
        """Serialize via dill."""
        (directory / "data.dill").write_bytes(dill.dumps(obj))

    @staticmethod
    def load(directory: Path) -> Any:
        """Deserialize via dill."""
        return dill.loads((directory / "data.dill").read_bytes())  # noqa: S301


class UnsupportedPayload:
    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UnsupportedPayload) and self.value == other.value

    __hash__ = object.__hash__


@meta(id="strict_task", cache=False)
def strict_task(config: object) -> int:
    _ = config
    return 1


@meta(id="strict_cached_task", cache=True)
def strict_cached_task(config: object) -> int:
    _ = config
    return 1


@meta(id="dependency_value", cache=False)
def dependency_value(seed: int) -> int:
    _ = seed
    return 1


@meta(id="consume_dependency", cache=False)
def consume_dependency(value: int) -> int:
    return value


@meta(id="hashable_result", cache=False)
def hashable_result(seed: int) -> int:
    _ = seed
    return 0


@meta(id="unsupported_result", cache=True, serializer=DillSerializer)
def unsupported_result(seed: int) -> UnsupportedPayload:
    return UnsupportedPayload(seed)


def test_task_rejects_unsupported_argument_with_guidance() -> None:
    with pytest.raises(HashError) as exc_info:
        Task(strict_task, config=lambda: None)

    message = str(exc_info.value)
    assert "Task 'strict_task' argument 'config' required unsupported hashing behavior." in message
    assert "explicit `stable_hash` handler" in message
    assert "Pass a `Task` dependency" in message
    assert "@meta(exclude=...)" in message
    assert "@meta(versions=...)" in message


def test_cacheable_task_rejects_unsupported_argument_with_stronger_guidance() -> None:
    with pytest.raises(HashError) as exc_info:
        Task(strict_cached_task, config=lambda: None)

    message = str(exc_info.value)
    assert "Cacheable task 'strict_cached_task' argument 'config' required unsupported hashing behavior." in message
    assert "Cache correctness depends on stable hashes." in message


def test_dependency_tasks_use_task_hash_for_graph_identity_and_result_hash_for_resolved_identity(
    tmp_path: Path,
) -> None:
    left_workspace = DiskWorkspace(directory=str(tmp_path / "left"))
    right_workspace = DiskWorkspace(directory=str(tmp_path / "right"))

    dependency = Task(dependency_value, seed=1)
    task_instance = Task(consume_dependency, value=dependency.T)
    graph_identity = task_instance.task_hash()

    left_workspace.set_result_hash(dependency, ResultHash.from_object("left-result"))
    right_workspace.set_result_hash(dependency, ResultHash.from_object("right-result"))

    assert task_instance.task_hash() == graph_identity
    assert task_instance.resolved_hash(workspace=left_workspace) != task_instance.resolved_hash(
        workspace=right_workspace
    )


def test_hashable_results_are_indexed_by_result(tmp_path: Path) -> None:
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))

    left = Task(hashable_result, seed=1)
    right = Task(hashable_result, seed=2)

    left.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True)
    right.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True)

    assert left.result_hash(workspace=workspace) == ResultHash.from_object(0)
    assert left.result_hash(workspace=workspace) == right.result_hash(workspace=workspace)


def test_unsupported_results_fall_back_to_resolved_task_identity(tmp_path: Path) -> None:
    workspace = DiskWorkspace(directory=str(tmp_path / ".misen"))

    task_instance = Task(unsupported_result, seed=7)
    value = task_instance.result(workspace=workspace, compute_if_uncached=True, compute_uncached_deps=True)

    assert value == UnsupportedPayload(7)
    assert task_instance.result_hash(workspace=workspace) == ResultHash.from_object(
        task_instance.resolved_hash(workspace)
    )
    assert task_instance.result(workspace=workspace) == UnsupportedPayload(7)


def test_public_task_api_no_longer_exposes_removed_hashing_options() -> None:
    task_signature = inspect.signature(meta)

    assert "argument_hash_policy" not in task_signature.parameters
    assert "index_by" not in task_signature.parameters
    assert "argument_hash_policy" not in TaskMetadata.__struct_fields__
    assert "index_by" not in TaskMetadata.__struct_fields__
