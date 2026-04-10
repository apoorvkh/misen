"""Task metadata model and public ``@task`` decorator.

This module defines the stable metadata contract used by :class:`misen.tasks.Task`:

- identity and cache behavior (`id`, `exclude`, `defaults`, `versions`)
- result persistence (`serializer`)
- execution requirements (`resources`)

The decorator writes this metadata onto function objects, while
``resolve_task_properties`` normalizes behavior for local functions, lambdas,
and external callables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeAlias, TypeVar

from msgspec import Struct

from misen.utils.function_introspection import (
    external_callable_id,
    is_lambda_function,
    is_local_project_function,
    lambda_task_id,
)
from misen.utils.hashing import ResultHash
from misen.utils.serde import DefaultSerializer, Serializer

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FunctionType

__all__ = ["GpuRuntime", "Resources", "TaskProperties", "resolve_task_properties", "task"]

P = ParamSpec("P")
R = TypeVar("R")


class TaskProperties(Struct, frozen=True):
    """Immutable metadata describing task identity, execution, and caching.

    Attributes:
        id: Stable task identifier.
        cache: Whether task results are persisted in the workspace.
        exclude: Argument names excluded from hash identity.
        defaults: Argument values treated as "default" and omitted from hashes
            when matching.
        versions: Per-argument hash-version overrides used to invalidate stale
            semantics without renaming the task.
        resources: Callable that computes resource requirements from arguments.
        serializer: Serializer type used to persist cached results.
        cleanup_work_dir: Whether to remove cacheable task work dirs after a
            successful run. Non-cacheable task work dirs are always cleaned up.
    """

    id: str
    cache: bool = False
    exclude: set[str] = set()
    defaults: dict[str, Any] = {}
    versions: dict[tuple[str, ResultHash], int] = {}
    resources: Callable[..., Resources] = lambda *_, **__: Resources()
    serializer: type[Serializer] = DefaultSerializer
    cleanup_work_dir: bool = False


GpuRuntime: TypeAlias = Literal["cuda", "rocm", "xpu"]


class Resources(Struct, frozen=True):
    """Resource requirements for executing a task.

    Attributes:
        time: Requested wall-clock time in minutes, if known.
        nodes: Number of nodes required.
        memory: Memory in GiB.
        cpus: CPU cores.
        gpus: GPU count.
        gpu_memory: Optional requested GPU memory in GiB.
        gpu_runtime: Requested GPU runtime.
    """

    time: int | None = None
    nodes: int = 1
    memory: int = 8
    cpus: int = 1
    gpus: int = 0
    gpu_memory: int | None = None
    gpu_runtime: GpuRuntime = "cuda"


def task(
    *,
    id: str | None = None,  # noqa: A002
    cache: bool = False,
    exclude: set[str] | None = None,
    defaults: dict[str, Any] | None = None,
    versions: dict[str, dict[Any, int]] | None = None,
    resources: Callable[..., Resources] | Resources | None = None,
    serializer: type[Serializer[R]] = DefaultSerializer,  # TODO: tighten generic serializer typing
    cleanup_work_dir: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Attach :class:`TaskProperties` metadata to a function.

    Args:
        id: Stable task identifier. Required for local project functions.
        cache: Whether task results should be stored in the workspace.
        exclude: Argument names excluded from task identity.
        defaults: Argument defaults excluded from task identity when equal.
        versions: Optional argument-value version map used to force hash
            changes for specific semantic revisions.
        resources: Static resources object or callable from function args to
            resources.
        serializer: Serializer class used for cached results.
        cleanup_work_dir: Whether to remove cacheable task work dirs after a
            successful run. Non-cacheable task work dirs are always cleaned up.

    Returns:
        A decorator that annotates the target function.

    Raises:
        ValueError: If ``id`` is not provided.
    """
    if id is None:
        msg = "id must be provided."
        raise ValueError(msg)

    if resources is None:
        resources = Resources()
    if isinstance(resources, Resources):
        # Normalize static resources into the callable shape expected by TaskProperties.
        resources = lambda r=resources, *_, **__: r  # noqa: E731

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        """Attach task metadata to the decorated function.

        Args:
            func: Function to annotate.

        Returns:
            The same function, now carrying ``__task_properties__``.
        """
        # Function objects are extended at runtime with metadata consumed by Task().
        func.__task_properties__ = TaskProperties(  # ty:ignore[unresolved-attribute]
            id=id,
            cache=cache,
            exclude=(exclude or set()),
            defaults=(defaults or {}),
            versions=_normalize_versions(versions=versions),
            resources=resources,
            serializer=serializer,
            cleanup_work_dir=cleanup_work_dir,
        )

        return func

    return decorator


def resolve_task_properties(func: FunctionType) -> TaskProperties:
    """Resolve :class:`TaskProperties` for a function object.

    Args:
        func: Python function object.

    Returns:
        Resolved task properties.

    Raises:
        ValueError: If a local project function lacks ``@task(...)`` metadata.
    """
    if is_lambda_function(func):
        return TaskProperties(lambda_task_id(func))

    if is_local_project_function(func):
        if not hasattr(func, "__task_properties__"):
            msg = (
                f"Local function {func.__module__}.{func.__qualname__} must define __task_properties__. Use @task(...)."
            )
            raise ValueError(msg)
        return func.__task_properties__

    return TaskProperties(external_callable_id(func))


def _normalize_versions(versions: dict[str, dict[Any, int]] | None) -> dict[tuple[str, ResultHash], int]:
    """Normalize argument-version mapping into hash-key lookup.

    Args:
        versions: Nested ``argument -> value -> version`` mapping.

    Returns:
        Flat mapping keyed by ``(argument_name, ResultHash(value))``.
    """
    return {
        (name, ResultHash.from_object(value)): version
        for name, value_to_version in (versions or {}).items()
        for value, version in value_to_version.items()
    }
