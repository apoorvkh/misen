"""Internal task helpers for traversal, hashing, and execution.

This module centralizes the mechanics used by :class:`misen.tasks.Task`:

- walking nested structures to find/map leaves
- canonical task-argument hashing
- runtime argument resolution (dependency results + sentinels)
- output capture and result persistence
"""

from __future__ import annotations

import itertools
import tempfile
import time
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from typing_extensions import assert_never

from misen.sentinels import ASSIGNED_RESOURCES, WORK_DIR
from misen.utils.hashes import ResultHash, TaskHash
from misen.utils.log_capture import capture_all_output
from misen.utils.runtime_events import runtime_event, task_label

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from inspect import Signature

    from misen.task_properties import TaskProperties
    from misen.tasks import Task
    from misen.utils.assigned_resources import AssignedResources
    from misen.workspace import Workspace

__all__ = [
    "collect_task_dependencies",
    "execute_task",
    "hash_task_arguments",
    "iter_nested_leaves",
    "map_nested_leaves",
    "save_task_result",
]

R = TypeVar("R")


def iter_nested_leaves(value: Any) -> Iterator[Any]:
    """Yield scalar leaves from supported nested containers.

    Args:
        value: Arbitrary nested structure.

    Yields:
        Non-container leaves.
    """
    if type(value) is dict:
        for key, nested in value.items():
            yield from iter_nested_leaves(key)
            yield from iter_nested_leaves(nested)
        return

    if type(value) is list or type(value) is set or type(value) is frozenset:
        for nested in value:
            yield from iter_nested_leaves(nested)
        return

    if type(value) is tuple:
        for nested in value:
            yield from iter_nested_leaves(nested)
        return

    yield value


def map_nested_leaves(value: Any, leaf_mapper: Callable[[Any], Any]) -> Any:
    """Map scalar leaves recursively while preserving container structure.

    Args:
        value: Arbitrary nested structure.
        leaf_mapper: Function applied to non-container leaves.

    Returns:
        Structure mirroring ``value`` with mapped leaves.
    """
    if type(value) is dict:
        mapped_items = [
            (
                map_nested_leaves(k, leaf_mapper),
                map_nested_leaves(v, leaf_mapper),
            )
            for k, v in value.items()
        ]
        return dict(mapped_items)
    if type(value) is list:
        return [map_nested_leaves(v, leaf_mapper) for v in value]
    if type(value) is tuple:
        return tuple(map_nested_leaves(v, leaf_mapper) for v in value)
    if type(value) is set:
        return {map_nested_leaves(v, leaf_mapper) for v in value}
    if type(value) is frozenset:
        return frozenset(map_nested_leaves(v, leaf_mapper) for v in value)
    return leaf_mapper(value)


def hash_task_arguments(
    *,
    signature: Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    properties: TaskProperties,
    hash_task_by_result: bool = False,
    workspace: Workspace | Literal["auto"] = "auto",
) -> dict[str, tuple[TaskHash | ResultHash, int]]:
    """Return canonical hashes for bound task arguments.

    Args:
        signature: Function signature used for canonical binding/defaults.
        args: Positional arguments.
        kwargs: Keyword arguments.
        properties: Task metadata controlling include/exclude/default/version.
        hash_task_by_result: Whether dependent tasks are represented by
            ``result_hash`` instead of ``task_hash``.
        workspace: Workspace used when hashing dependencies by result.

    Returns:
        Mapping ``argument_name -> (hash_value, version)``.

    Raises:
        RuntimeError: If sentinel values appear in arguments during hash
            calculation.
    """
    from misen.tasks import Task

    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()

    def leaf_representation(value: Any) -> TaskHash | ResultHash | Any:
        if isinstance(value, Task):
            return value.result_hash(workspace=workspace) if hash_task_by_result else value.task_hash()
        return value

    def argument_hash(value: Any) -> TaskHash | ResultHash:
        if isinstance(value, Task):
            return cast("TaskHash | ResultHash", leaf_representation(value))

        if value is ASSIGNED_RESOURCES or value is WORK_DIR:
            msg = "Resolved task arguments cannot contain sentinel values."
            raise RuntimeError(msg)

        return ResultHash.from_object(map_nested_leaves(value, leaf_representation))

    def include_argument(name: str, value: Any) -> bool:
        return name not in properties.exclude and (
            name not in properties.defaults or properties.defaults[name] != value
        )

    hashed_arguments: dict[str, tuple[TaskHash | ResultHash, int]] = {}

    for name, value in bound_arguments.arguments.items():
        if not include_argument(name, value):
            continue
        arg_hash = argument_hash(value)
        version = properties.versions.get((name, cast("ResultHash", arg_hash)), 0)
        hashed_arguments[name] = (arg_hash, version)

    return hashed_arguments


def collect_task_dependencies(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> frozenset[Task[Any]]:
    """Collect task dependencies nested within args/kwargs.

    Args:
        args: Positional argument tuple.
        kwargs: Keyword argument mapping.

    Returns:
        Frozen set of discovered dependent tasks.
    """
    from misen.tasks import Task

    values = itertools.chain(args, kwargs.values())
    leaves = itertools.chain.from_iterable(map(iter_nested_leaves, values))
    return frozenset(leaf for leaf in leaves if isinstance(leaf, Task))


def execute_task(
    task: Task[R],
    workspace: Workspace,
    dependency_results: dict[Task[Any], Any],
    assigned_resources: AssignedResources | None,
    job_id: str | None,
) -> R:
    """Execute task function under log capture.

    Args:
        task: Task to execute.
        workspace: Workspace for logs/artifacts.
        dependency_results: Precomputed dependency results.
        assigned_resources: Optional runtime resources for sentinel injection.
        job_id: Optional job id for task-log grouping.

    Returns:
        Task result value.
    """
    task_name = task_label(task)
    runtime_event(f"Task started: {task_name} (job_id={job_id or 'n/a'})", style="yellow")
    started_at = time.perf_counter()

    argument_resolver = _build_argument_resolver(
        task=task,
        workspace=workspace,
        dependency_results=dependency_results,
        assigned_resources=assigned_resources,
    )

    try:
        with workspace.open_task_log(task=task, mode="a", job_id=job_id, timestamp="current") as task_log:
            with capture_all_output(task_log, tee_to_stdout=True):
                args = (argument_resolver(value) for value in task.args)
                kwargs = {name: argument_resolver(value) for name, value in task.kwargs.items()}
                result = task.func(*args, **kwargs)
    except Exception:
        runtime_event(
            f"Task failed: {task_name} in {(time.perf_counter() - started_at):.2f}s",
            style="bold red",
        )
        raise

    runtime_event(f"Task finished: {task_name} in {(time.perf_counter() - started_at):.2f}s", style="green")
    return cast("R", result)


def save_task_result(task: Task[Any], result: Any, workspace: Workspace) -> None:
    """Persist task result metadata and optional cached payload.

    Args:
        task: Executed task.
        result: Computed result.
        workspace: Workspace to update.
    """
    match task.properties.index_by:
        case "task":
            index = task.resolved_hash(workspace=workspace)
        case "result":
            index = result
        case _:
            assert_never(task.properties.index_by)

    workspace.set_result_hash(task, ResultHash.from_object(index))

    if task.properties.cache:
        try:
            workspace.results[task] = result
        except Exception:
            del workspace.results[task]
            workspace.clear_result_hash(task=task)
            raise


def _build_argument_resolver(
    task: Task[Any],
    workspace: Workspace,
    dependency_results: dict[Task[Any], Any],
    assigned_resources: AssignedResources | None,
) -> Callable[[Any], Any]:
    """Build argument resolver for runtime task execution.

    Args:
        task: Task being executed.
        workspace: Workspace providing work-dir and cached dependency access.
        dependency_results: Immediate dependency result map.
        assigned_resources: Optional runtime resource assignment.

    Returns:
        Callable that maps arbitrary nested argument structures into runtime
        values (dependency outputs, work dirs, assigned resources).
    """
    from misen.tasks import Task

    @cache
    def work_dir() -> Path:
        """Return cache-backed or temporary work directory for this execution."""
        if task.properties.cache:
            return workspace.get_work_dir(task=task)

        resolved = task.resolved_hash(workspace=workspace).b32()
        return Path(tempfile.mkdtemp(prefix=f"misen-work-{resolved}-"))

    def argument_resolver(value: Any) -> Any:
        def leaf_resolver(leaf: Any) -> Any:
            if isinstance(leaf, Task):
                return dependency_results[leaf]
            if leaf is WORK_DIR:
                return work_dir()
            if leaf is ASSIGNED_RESOURCES:
                return assigned_resources
            return leaf

        return map_nested_leaves(value, leaf_resolver)

    return argument_resolver
