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
import time
import traceback
from contextlib import contextmanager, suppress
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from misen.exceptions import ExecutionError, HashError, StorageError
from misen.sentinels import SCRATCH_DIR, is_runtime_sentinel
from misen.task_metadata import aggregate_resources
from misen.utils.cli import system_exit_code
from misen.utils.graph import DependencyGraph
from misen.utils.hashing import ResultHash, TaskHash
from misen.utils.log_capture import capture_all_output
from misen.utils.nested import iter_nested_leaves, map_nested_leaves
from misen.utils.runtime_events import runtime_event, task_label

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from inspect import Signature
    from pathlib import Path
    from types import BuiltinFunctionType, FunctionType
    from typing import TextIO

    from misen.task_metadata import TaskMetadata
    from misen.tasks import Task
    from misen.utils.runtime_values import RuntimeValues
    from misen.workspace import Workspace

__all__ = [
    "build_task_dependency_graph",
    "collect_task_dependencies",
    "execute_task",
    "hash_task_arguments",
    "save_task_result",
    "validate_task_sentinels",
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

    Notes:
        Arguments bound to runtime sentinels (e.g. ``SCRATCH_DIR``) are
        excluded from the result: the injected value varies per
        workspace/machine and must not contribute to task identity.
    """
    from misen.tasks import Task

    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()

    def leaf_representation(value: Any) -> TaskHash | ResultHash | Any:
        if isinstance(value, Task):
            return value.result_hash(workspace=workspace) if hash_task_by_result else value.task_hash()
        return value

    def include_argument(name: str, value: Any) -> bool:
        if is_runtime_sentinel(value):
            return False
        return name not in meta.exclude and (name not in meta.defaults or meta.defaults[name] != value)

    hashed_arguments: dict[str, tuple[TaskHash | ResultHash, int]] = {}

    for name, value in bound_arguments.arguments.items():
        if not include_argument(name, value):
            continue
        try:
            arg_hash = (
                cast("TaskHash | ResultHash", leaf_representation(value))
                if isinstance(value, Task)
                else ResultHash.from_object(map_nested_leaves(value, leaf_representation))
            )
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


def validate_task_sentinels(
    *,
    signature: Signature,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    func: FunctionType | BuiltinFunctionType,
) -> None:
    """Reject runtime-sentinel misuse when a task is constructed.

    Sentinels (e.g. ``SCRATCH_DIR``) are resolved by mapping over a task's
    bound ``args``/``kwargs`` at execution time, so they are only supported as
    top-level ``Task(...)`` arguments. Two misuses would otherwise leak the
    raw sentinel object into the function body mid-run:

    - a sentinel used as a *function-signature default* that is not explicitly
      bound at ``Task(...)`` — Python applies signature defaults at call time,
      after argument resolution, so the resolver never sees it
    - a sentinel *nested inside a container* argument — the resolver replaces
      only top-level values

    Args:
        signature: Task function signature.
        args: Positional arguments bound at ``Task(...)``.
        kwargs: Keyword arguments bound at ``Task(...)``.
        func: Task function, used for error messages.

    Raises:
        TypeError: If an unbound parameter has a sentinel signature default,
            or a sentinel appears nested inside a container argument.
    """
    bound_names = set(signature.bind_partial(*args, **kwargs).arguments)
    for name, parameter in signature.parameters.items():
        if name in bound_names or not is_runtime_sentinel(parameter.default):
            continue
        msg = (
            f"{parameter.default!r} cannot be a function-signature default (parameter '{name}' of "
            f"{func.__name__}): Python applies signature defaults at call time, bypassing misen's "
            f"argument resolver, so the raw sentinel object would reach the function body. Keep the "
            f"signature misen-agnostic (e.g. '{name}: Path') and bind the sentinel when constructing "
            f"the task: Task({func.__name__}, {name}={parameter.default!r})."
        )
        raise TypeError(msg)

    named_values = itertools.chain(
        ((f"positional argument {index}", value) for index, value in enumerate(args)),
        ((f"argument '{name}'", value) for name, value in kwargs.items()),
    )
    for location, value in named_values:
        if is_runtime_sentinel(value):
            continue
        nested = next((leaf for leaf in iter_nested_leaves(value) if is_runtime_sentinel(leaf)), None)
        if nested is not None:
            msg = (
                f"{nested!r} must be a top-level Task(...) argument, but it is nested inside "
                f"{location} of {func.__name__}. Nested sentinels are not resolved at runtime; "
                f"pass the sentinel directly instead, e.g. Task({func.__name__}, ..., "
                f"my_param={nested!r})."
            )
            raise TypeError(msg)


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
    merged: dict[Task[Any], Task[Any]] = {}
    for task in (leaf for leaf in leaves if isinstance(leaf, Task)):
        existing = merged.get(task)
        merged[task] = task if existing is None else _merge_equivalent_task(existing, task)
    return frozenset(merged.values())


def _merge_equivalent_task(existing: Task[Any], candidate: Task[Any]) -> Task[Any]:
    """Merge scheduler-facing resources for two task-equal handles."""
    merged_resources = aggregate_resources((existing.resources, candidate.resources), sum_time=False)
    return existing if merged_resources == existing.resources else existing.with_resources(**merged_resources)


def execute_task(
    task: Task[R],
    workspace: Workspace,
    dependency_results: dict[Task[Any], Any],
    job_id: str,
    log_task: Task[Any] | None = None,
    *,
    scratch_dir: Path | None = None,
    runtime_values: RuntimeValues | None = None,
) -> R:
    """Execute task function under log capture.

    Args:
        task: Task to execute.
        workspace: Workspace for logs/artifacts.
        dependency_results: Precomputed dependency results.
        job_id: Job id for task-log grouping.
        log_task: Task identity to use for log storage. This can differ from
            ``task`` when a work unit executes an internal copy with resolved
            non-cacheable dependencies.
        scratch_dir: Pre-created scratch directory for this run, or ``None``
            if the task does not request one. The caller owns lifecycle:
            cleanup on success and (for non-cacheable tasks) on failure
            happens in :meth:`misen.tasks.Task.result`.
        runtime_values: Work-unit-scoped runtime values used to resolve
            sentinels other than ``SCRATCH_DIR``.

    Returns:
        The task's result value.
    """
    argument_resolver = _build_argument_resolver(
        dependency_results=dependency_results,
        scratch_dir=scratch_dir,
        runtime_values=runtime_values,
    )

    resolved_args = tuple(argument_resolver(value) for value in task.args)
    resolved_kwargs = {name: argument_resolver(value) for name, value in task.kwargs.items()}
    display = _format_resolved_call(task, resolved_args, resolved_kwargs)
    debug_name = task_label(task)

    logger.info("Task started: %s (job_id=%s).", debug_name, job_id)
    runtime_event(f"Task started: {display}", style="yellow")
    started_at = time.perf_counter()

    log_identity = log_task or task
    log_path = workspace.get_task_log(task=log_identity, job_id=job_id)
    sync_scratch_dir = task.meta.cache and scratch_dir is not None
    primary_error: BaseException | None = None
    try:
        if sync_scratch_dir:
            workspace.start_scratch_dir_sync(task=task)
        with _open_task_log(log_path) as task_log:
            try:
                with capture_all_output(task_log, tee_to_stdout=True):
                    result = task.func(*resolved_args, **resolved_kwargs)
            except BaseException as exc:
                # Wait for the capture reader to drain before appending the
                # traceback, so buffered stdout/stderr cannot appear after it.
                try:
                    traceback.print_exception(exc, file=task_log)
                except Exception as log_error:  # noqa: BLE001 -- diagnostics are best-effort
                    exc.add_note(f"Additionally, writing the task traceback failed: {log_error}")
                raise
    except BaseException as exc:
        error: BaseException = exc
        if isinstance(exc, SystemExit):
            exit_code = system_exit_code(exc)
            error = ExecutionError(
                f"Task {debug_name} requested interpreter exit with code {exit_code} before completing."
            )
        primary_error = error
        elapsed_s = time.perf_counter() - started_at
        error.add_note(f"Misen task {debug_name} failed (job_id={job_id}, log={log_path}).")
        logger.error("Task failed: %s after %.2fs.", debug_name, elapsed_s)  # noqa: TRY400 -- rendered upstream
        with suppress(Exception, SystemExit):
            runtime_event(f"Task failed: {display} in {elapsed_s:.2f}s", style="bold red")
        if error is exc:
            raise
        raise error from exc
    finally:
        cleanups: list[tuple[str, Callable[[], None]]] = []
        if sync_scratch_dir:
            cleanups.append(("scratch-directory sync", lambda: workspace.finalize_scratch_dir(task=task)))
        cleanups.append(("task-log sync", lambda: workspace.finalize_task_log(task=log_identity, job_id=job_id)))
        _run_task_cleanups(primary_error, cleanups)

    logger.debug("Task function returned: %s.", debug_name)
    return cast("R", result)


@contextmanager
def _open_task_log(path: Path) -> Iterator[TextIO]:
    """Open a Misen-owned task log without masking an active task failure."""
    try:
        task_log = path.open("a", buffering=1, encoding="utf-8")
    except OSError as exc:
        msg = f"Could not open task log {path}: {exc}"
        raise StorageError(msg) from exc

    primary_error: BaseException | None = None
    try:
        yield task_log
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            task_log.close()
        except OSError as exc:
            if primary_error is None:
                msg = f"Could not close task log {path}: {exc}"
                raise StorageError(msg) from exc
            primary_error.add_note(f"Additionally, closing task log {path} failed: {type(exc).__name__}: {exc}")


def _run_task_cleanups(
    primary_error: BaseException | None,
    cleanups: list[tuple[str, Callable[[], None]]],
) -> None:
    """Run every finalizer without replacing an active task failure."""
    error = primary_error
    for label, cleanup in cleanups:
        try:
            cleanup()
        except BaseException as exc:
            if error is None:
                error = exc
            else:
                error.add_note(f"Additional error during {label}: {type(exc).__name__}: {exc}")
                logger.exception("Additional error during %s; preserving the earlier failure.", label)
    if primary_error is None and error is not None:
        raise error


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
        if len(text) > 80:  # noqa: PLR2004
            text = text[:77] + "..."
        parts.append(f"{name}={text}")
    return f"{task.func.__name__}({', '.join(parts)})" if parts else f"{task.func.__name__}()"


def save_task_result(
    task: Task[Any],
    result: Any,
    workspace: Workspace,
    *,
    check_ownership: Callable[[], None] | None = None,
) -> None:
    """Persist a task payload before its hash pointer, fencing both commits."""
    try:
        result_hash = ResultHash.from_object(result)
        index_mode = "result"
    except HashError:
        result_hash = ResultHash.from_object(task.resolved_hash(workspace=workspace))
        index_mode = "task"

    logger.debug("Persisting result hash for %s using index_mode=%s.", task, index_mode)

    # Write the payload before its pointer so crashes cannot publish missing data.
    if task.meta.cache:
        workspace.results.store(task, result, result_hash, before_commit=check_ownership)

    workspace.set_result_hash(task, result_hash, before_commit=check_ownership)


def _build_argument_resolver(
    dependency_results: dict[Task[Any], Any],
    *,
    scratch_dir: Path | None,
    runtime_values: RuntimeValues | None,
) -> Callable[[Any], Any]:
    """Build argument resolver for runtime task execution.

    Args:
        dependency_results: Immediate dependency result map.
        scratch_dir: Pre-created scratch directory if the task requested
            one via ``SCRATCH_DIR``; ``None`` otherwise.
        runtime_values: Work-unit-scoped values for other sentinels.

    Returns:
        Callable mapping arbitrary nested argument structures into runtime
        values (dependency outputs, scratch dirs).
    """
    from misen.tasks import Task

    def argument_resolver(value: Any) -> Any:
        if value is SCRATCH_DIR:
            if scratch_dir is None:
                msg = "SCRATCH_DIR sentinel resolved but no scratch directory was provided to execute_task."
                raise ExecutionError(msg)
            return scratch_dir
        if is_runtime_sentinel(value):
            if runtime_values is None:
                msg = f"{value!r} can only be resolved while executing a WorkUnit through an Executor."
                raise ExecutionError(msg)
            return runtime_values.resolve(value)
        return map_nested_leaves(
            value,
            lambda leaf: dependency_results[leaf] if isinstance(leaf, Task) else leaf,
        )

    return argument_resolver


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
    if exclude_cached and not exclude_cacheable and workspace is None:
        msg = "workspace is required when exclude_cached=True."
        raise ValueError(msg)

    if exclude_cacheable or exclude_cached:

        @cache
        def include_dependency(dependency: Task[Any]) -> bool:
            if exclude_cacheable:
                return not dependency.meta.cache
            return not dependency.is_cached(workspace=cast("Workspace", workspace))

    else:

        def include_dependency(dependency: Task[Any]) -> bool:  # noqa: ARG001
            return True

    graph: DependencyGraph[Task[Any]] = DependencyGraph()
    nodes: dict[Task[Any], int] = {}

    def get_node_index(candidate: Task[Any]) -> tuple[int, bool]:
        node_index = nodes.get(candidate)
        if node_index is None:
            node_index = nodes[candidate] = graph.add_node(candidate)
            return node_index, True
        graph[node_index] = _merge_equivalent_task(graph[node_index], candidate)
        return node_index, False

    stack: list[Task[Any]] = [task]

    while stack:
        current = stack.pop()
        current_node, _ = get_node_index(current)

        for dependency in current.dependencies:
            if not include_dependency(dependency):
                continue
            dependency_node, is_new = get_node_index(dependency)
            graph.add_edge(current_node, dependency_node)
            if is_new:
                stack.append(dependency)

    return graph
