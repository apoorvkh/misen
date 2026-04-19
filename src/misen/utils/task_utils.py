"""Internal task helpers for hashing, execution, and persistence.

This module centralizes the mechanics used by :class:`misen.tasks.Task`:

- canonical task-argument hashing
- runtime argument resolution (dependency results + sentinels)
- output capture and result persistence

Generic nested-structure traversal lives in :mod:`misen.utils.nested`.
"""

from __future__ import annotations

import itertools
import logging
import tempfile
import time
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from rich.console import Console as RichConsole

from misen.exceptions import HashError
from misen.sentinels import ASSIGNED_RESOURCES, ASSIGNED_RESOURCES_PER_NODE, WORK_DIR
from misen.utils.graph import DependencyGraph
from misen.utils.hashing import ResultHash, TaskHash
from misen.utils.log_capture import capture_all_output
from misen.utils.nested import iter_nested_leaves, map_nested_leaves
from misen.utils.runtime_events import runtime_event, task_label

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from inspect import Signature

    from misen.task_metadata import TaskMetadata
    from misen.tasks import Task
    from misen.utils.assigned_resources import AssignedResources, AssignedResourcesPerNode
    from misen.workspace import Workspace

__all__ = [
    "build_task_dependency_graph",
    "collect_task_dependencies",
    "execute_task",
    "hash_task_arguments",
    "save_task_result",
]

R = TypeVar("R")
logger = logging.getLogger(__name__)


def hash_task_arguments(
    *,
    signature: Signature,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    meta: TaskMetadata,
    hash_task_by_result: bool = False,
    workspace: Workspace | Literal["auto"] = "auto",
) -> dict[str, tuple[TaskHash | ResultHash, int]]:
    """Return canonical hashes for bound task arguments.

    Args:
        signature: Function signature used for canonical binding/defaults.
        args: Positional arguments.
        kwargs: Keyword arguments.
        meta: Task metadata controlling include/exclude/default/version.
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

        if value is ASSIGNED_RESOURCES or value is ASSIGNED_RESOURCES_PER_NODE or value is WORK_DIR:
            msg = "Resolved task arguments cannot contain sentinel values."
            raise RuntimeError(msg)

        return ResultHash.from_object(map_nested_leaves(value, leaf_representation))

    def include_argument(name: str, value: Any) -> bool:
        return name not in meta.exclude and (
            name not in meta.defaults or meta.defaults[name] != value
        )

    hashed_arguments: dict[str, tuple[TaskHash | ResultHash, int]] = {}

    for name, value in bound_arguments.arguments.items():
        if not include_argument(name, value):
            continue
        try:
            arg_hash = argument_hash(value)
        except HashError as exc:
            prefix = f"Task '{meta.id}' argument '{name}' required unsupported hashing behavior. "
            if meta.cache:
                prefix = (
                    f"Cacheable task '{meta.id}' argument '{name}' required unsupported hashing behavior. "
                    "Cache correctness depends on stable hashes. "
                )

            msg = (
                f"{prefix}Non-Task argument values must hash through an explicit `stable_hash` handler. "
                f"Details: {exc} "
                "Pass a `Task` dependency, register a `stable_hash` handler, or use "
                "`@meta(exclude=...)` / `@meta(versions=...)`."
            )
            raise HashError(msg) from exc
        version = meta.versions.get((name, cast("ResultHash", arg_hash)), 0)
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
    assigned_resources: AssignedResources | AssignedResourcesPerNode | None,
    job_id: str,
) -> tuple[R, Path | None]:
    """Execute task function under log capture.

    Args:
        task: Task to execute.
        workspace: Workspace for logs/artifacts.
        dependency_results: Precomputed dependency results.
        assigned_resources: Optional runtime resources for sentinel injection.
        job_id: Job id for task-log grouping.

    Returns:
        Task result value plus the runtime work directory used (if any).
    """
    argument_resolver, work_directory = _build_argument_resolver(
        task=task,
        workspace=workspace,
        dependency_results=dependency_results,
        assigned_resources=assigned_resources,
    )

    resolved_args = tuple(argument_resolver(value) for value in task.args)
    resolved_kwargs = {name: argument_resolver(value) for name, value in task.kwargs.items()}
    display = _format_resolved_call(task, resolved_args, resolved_kwargs)
    debug_name = task_label(task)

    logger.info("Task started: %s (job_id=%s).", debug_name, job_id)
    runtime_event(f"Task started: {display}", style="yellow")
    started_at = time.perf_counter()

    with workspace.open_task_log(task=task, mode="a", job_id=job_id) as task_log:
        with capture_all_output(task_log, tee_to_stdout=True):
            try:
                result = task.func(*resolved_args, **resolved_kwargs)
            except Exception:
                RichConsole(stderr=True).print_exception()
                logger.exception("Task failed: %s after %.2fs.", debug_name, time.perf_counter() - started_at)
                runtime_event(
                    f"Task failed: {display} in {(time.perf_counter() - started_at):.2f}s",
                    style="bold red",
                )
                raise

    logger.info("Task finished: %s in %.2fs.", debug_name, time.perf_counter() - started_at)
    runtime_event(f"Task finished: {display} in {(time.perf_counter() - started_at):.2f}s", style="green")
    return cast("R", result), work_directory


def _format_resolved_call(task: Task[Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Return ``name(arg=value, ...)`` using resolved runtime argument values.

    Argument values are ``repr``'d and each individual rendering is capped at
    80 characters to keep runtime messages readable when an argument is a
    large object or a long path.
    """
    bound = task._signature.bind_partial(*args, **kwargs)  # noqa: SLF001
    parts: list[str] = []
    for name, value in bound.arguments.items():
        if name in task.meta.exclude:
            continue
        text = repr(value)
        if len(text) > 80:
            text = text[:77] + "..."
        parts.append(f"{name}={text}")
    return f"{task.meta.id}({', '.join(parts)})" if parts else f"{task.meta.id}()"


def save_task_result(task: Task[Any], result: Any, workspace: Workspace) -> None:
    """Persist task result metadata and optional cached payload.

    Args:
        task: Executed task.
        result: Computed result.
        workspace: Workspace to update.
    """
    try:
        result_hash = ResultHash.from_object(result)
        index_mode = "result"
    except HashError:
        result_hash = ResultHash.from_object(task.resolved_hash(workspace=workspace))
        index_mode = "task"

    logger.debug("Persisting result hash for %s using index_mode=%s.", task, index_mode)

    workspace.set_result_hash(task, result_hash)

    if task.meta.cache:
        try:
            workspace.results[task] = result
        except Exception:
            logger.exception("Failed while persisting cached result payload for %s; rolling back hash metadata.", task)
            del workspace.results[task]
            workspace.clear_result_hash(task=task)
            raise


def _build_argument_resolver(
    task: Task[Any],
    workspace: Workspace,
    dependency_results: dict[Task[Any], Any],
    assigned_resources: AssignedResources | AssignedResourcesPerNode | None,
) -> tuple[Callable[[Any], Any], Path | None]:
    """Build argument resolver for runtime task execution.

    Args:
        task: Task being executed.
        workspace: Workspace providing work-dir and cached dependency access.
        dependency_results: Immediate dependency result map.
        assigned_resources: Optional runtime resource assignment (single-node or multi-node).

    Returns:
        Tuple of:
        - callable mapping arbitrary nested argument structures into runtime
          values (dependency outputs, work dirs, assigned resources)
        - runtime work directory if ``WORK_DIR`` is requested
    """
    from misen.tasks import Task

    work_directory: Path | None = None

    def work_dir() -> Path:
        """Return cache-backed or temporary work directory for this execution."""
        nonlocal work_directory
        if work_directory is None:
            if task.meta.cache:
                work_directory = workspace.get_work_dir(task=task)
            else:
                resolved = task.resolved_hash(workspace=workspace).b32()
                work_directory = Path(tempfile.mkdtemp(prefix=f"misen-work-{resolved}-"))
        return work_directory

    def argument_resolver(value: Any) -> Any:
        if value is WORK_DIR:
            return work_dir()
        if value is ASSIGNED_RESOURCES or value is ASSIGNED_RESOURCES_PER_NODE:
            return assigned_resources
        return map_nested_leaves(
            value,
            lambda leaf: dependency_results[leaf] if isinstance(leaf, Task) else leaf,
        )

    if WORK_DIR in itertools.chain(task.args, task.kwargs.values()):
        work_dir()

    return argument_resolver, work_directory


def build_task_dependency_graph(
    task: Task[Any],
    *,
    exclude_cacheable: bool = False,
    exclude_cached: bool = False,
    workspace: Workspace | None = None,
) -> DependencyGraph[Task[Any]]:
    """Build dependency graph rooted at a task-like object.

    Args:
        task: Root task-like node.
        exclude_cacheable: Whether to skip cacheable dependency nodes.
        exclude_cached: Whether to skip dependencies already cached in workspace.
        workspace: Workspace required when ``exclude_cached=True``.

    Returns:
        Dependency graph with edges ``task -> dependency``.

    Raises:
        ValueError: If ``exclude_cached=True`` and workspace is not provided.
    """
    if exclude_cacheable:

        @cache
        def include_dependency(dependency: Task[Any]) -> bool:
            return dependency.meta.cache is False

    elif exclude_cached:
        if workspace is None:
            msg = "workspace is required when exclude_cached=True."
            raise ValueError(msg)

        @cache
        def include_dependency(dependency: Task[Any]) -> bool:
            return not dependency.is_cached(workspace=workspace)

    else:

        def include_dependency(dependency: Task[Any]) -> bool:  # noqa: ARG001
            return True

    graph: DependencyGraph[Task[Any]] = DependencyGraph()
    nodes: dict[Task[Any], int] = {}

    def get_node_index(candidate: Task[Any]) -> int:
        node_index = nodes.get(candidate)
        if node_index is None:
            node_index = nodes[candidate] = graph.add_node(candidate)
        return node_index

    stack: list[Task[Any]] = [task]
    seen: set[Task[Any]] = {task}

    while stack:
        current = stack.pop()
        current_node = get_node_index(current)

        for dependency in current.dependencies:
            if not include_dependency(dependency):
                continue
            graph.add_edge(current_node, get_node_index(dependency), None)
            if dependency not in seen:
                seen.add(dependency)
                stack.append(dependency)

    return graph
