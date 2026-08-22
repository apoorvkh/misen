"""Task metadata model and public ``@meta`` decorator.

This module defines the stable metadata contract used by :class:`misen.tasks.Task`:

- identity and cache behavior (`id`, `exclude`, `defaults`, `versions`)
- result persistence (`serializer`)
- execution requirements (`resources`)

The decorator writes this metadata onto function objects, while
``resolve_task_metadata`` normalizes behavior for local functions, lambdas,
and external callables.
"""

from __future__ import annotations

from types import FunctionType
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeAlias, TypedDict, TypeVar, cast

from msgspec import Struct

from misen.utils.function_introspection import (
    external_callable_id,
    is_lambda_function,
    is_local_project_function,
    lambda_task_id,
)
from misen.utils.hashing import ResultHash

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import BuiltinFunctionType

    from misen.utils.serde import Serializer

__all__ = [
    "AcceleratorType",
    "Resources",
    "TaskMetadata",
    "aggregate_resources",
    "meta",
    "resolve_task_metadata",
]

P = ParamSpec("P")
R = TypeVar("R")
AcceleratorType: TypeAlias = Literal["cuda", "rocm", "xpu", "mps", "tpu"]


class Resources(TypedDict, total=False):
    """Resource requirements for executing a task.

    Attributes:
        time: Requested wall-clock time in minutes.
        memory: Memory in GiB per node.
        cpus: CPU cores per node.
        nodes: Compute nodes.
        accelerators: Accelerator devices per node.
        accelerator_type: Concrete accelerator backend.
        accelerator_memory: Minimum memory in GiB per device.
    """

    time: int
    memory: int
    cpus: int
    nodes: int
    accelerators: int
    accelerator_type: AcceleratorType
    accelerator_memory: int | None


_DEFAULT_RESOURCES: Resources = {
    "time": 60,
    "memory": 8,
    "cpus": 1,
    "nodes": 1,
    "accelerators": 0,
    "accelerator_type": "cuda",
    "accelerator_memory": None,
}


def _normalize_resources(resources: Resources) -> Resources:
    """Fill defaults and validate a resource request."""
    unknown = set(resources) - Resources.__annotations__.keys()
    if unknown:
        msg = f"Unknown Resources fields: {sorted(unknown)}"
        raise TypeError(msg)

    normalized = cast("Resources", {**_DEFAULT_RESOURCES, **resources})
    for name in ("time", "memory", "cpus", "nodes"):
        value = normalized[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            msg = f"{name} must be a positive integer."
            raise ValueError(msg)
    if normalized["accelerator_type"] not in ("cuda", "rocm", "xpu", "mps", "tpu"):
        msg = f"Unsupported accelerator type: {normalized['accelerator_type']!r}."
        raise ValueError(msg)
    count = normalized["accelerators"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        msg = "accelerators must be a nonnegative integer."
        raise ValueError(msg)
    memory = normalized["accelerator_memory"]
    if memory is not None and (isinstance(memory, bool) or not isinstance(memory, int) or memory < 1):
        msg = "accelerator_memory must be None or a positive integer number of GiB per device."
        raise ValueError(msg)
    if count == 0 and memory is not None:
        msg = "accelerator_memory requires accelerators > 0."
        raise ValueError(msg)
    return normalized


def aggregate_resources(resources: Iterable[Resources], *, sum_time: bool = True) -> Resources:
    """Combine multiple resource requests into one conservative request.

    Node, CPU, memory, and accelerator counts use ``max``. Accelerator-using
    tasks must agree on their type.

    Args:
        resources: Resource requests to normalize and merge.
        sum_time: Sum wall-clock limits for sequential work, or take their
            maximum when merging alternative handles for the same task.

    Returns:
        A single :class:`Resources` that satisfies every input request.

    Raises:
        TypeError: If a request contains an unknown field.
        ValueError: If the iterable is empty or active accelerator types conflict.
    """
    resource_list = [_normalize_resources(resource) for resource in resources]
    if not resource_list:
        msg = "aggregate_resources requires at least one Resources instance."
        raise ValueError(msg)
    accelerator_types = sorted({r["accelerator_type"] for r in resource_list if r["accelerators"] > 0})
    if len(accelerator_types) > 1:
        msg = f"Incompatible accelerator types: {accelerator_types}."
        raise ValueError(msg)
    accelerator_memories = [r["accelerator_memory"] for r in resource_list if r["accelerator_memory"] is not None]

    return Resources(
        time=(sum if sum_time else max)(r["time"] for r in resource_list),
        memory=max(r["memory"] for r in resource_list),
        cpus=max(r["cpus"] for r in resource_list),
        nodes=max(r["nodes"] for r in resource_list),
        accelerators=max(r["accelerators"] for r in resource_list),
        accelerator_type=accelerator_types[0] if accelerator_types else "cuda",
        accelerator_memory=max(accelerator_memories) if accelerator_memories else None,
    )


class TaskMetadata(Struct, frozen=True):
    """Immutable metadata describing task identity, execution, and caching.

    Attributes:
        id: Stable task identifier. An empty string represents an unresolved
            decorator placeholder and is rejected when constructing a task.
        cache: Whether task results are persisted in the workspace.
        exclude: Argument names excluded from hash identity.
        defaults: Argument values treated as "default" and omitted from hashes
            when matching.
        versions: Per-argument hash-version overrides used to invalidate stale
            semantics without renaming the task.
        resources: Callable that computes resource requirements from arguments.
        serializer: Serializer type used to persist cached results.
    """

    id: str
    cache: bool = False
    exclude: set[str] = set()
    defaults: dict[str, Any] = {}
    versions: dict[tuple[str, ResultHash], int] = {}
    resources: Callable[..., Resources] = lambda *_, **__: _DEFAULT_RESOURCES
    serializer: type[Serializer] | None = None

    def resolve_resources(self, *args: Any, **kwargs: Any) -> Resources:
        """Compute resource requirements for this task, merging with defaults."""
        return _normalize_resources(self.resources(*args, **kwargs))


def meta(
    *,
    id: str = "",  # noqa: A002
    cache: bool = False,
    exclude: set[str] | None = None,
    defaults: dict[str, Any] | None = None,
    versions: dict[str, dict[Any, int]] | None = None,
    resources: Callable[..., Resources] | Resources | None = None,
    serializer: type[Serializer[R]] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Attach :class:`TaskMetadata` metadata to a function.

    Args:
        id: Stable task identifier. An omitted or empty string is a temporary
            placeholder that must be filled before constructing a
            :class:`misen.tasks.Task`.
        cache: Whether task results should be stored in the workspace.
        exclude: Argument names excluded from task identity.
        defaults: Argument defaults excluded from task identity when equal.
        versions: Optional argument-value version map used to force hash
            changes for specific semantic revisions.
        resources: Static resources object or callable from function args to
            resources.
        serializer: Serializer class used for cached results.

    Returns:
        A decorator that annotates the target function.
    """
    resources_fn: Callable[..., Resources]
    if resources is None or isinstance(resources, dict):
        resources_fn = lambda *_, r=resources, **__: cast("Resources", r or Resources())  # noqa: E731
    else:
        resources_fn = cast("Callable[..., Resources]", resources)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        """Attach task metadata to the decorated function.

        Args:
            func: Function to annotate.

        Returns:
            The same function, now carrying ``__task_metadata__``.
        """
        # Function objects are extended at runtime with metadata consumed by Task().
        func.__task_metadata__ = TaskMetadata(  # ty:ignore[unresolved-attribute]
            id=id,
            cache=cache,
            exclude=(exclude or set()),
            defaults=(defaults or {}),
            versions=_normalize_versions(versions=versions),
            resources=resources_fn,
            serializer=serializer,
        )

        return func

    return decorator


def resolve_task_metadata(func: FunctionType | BuiltinFunctionType) -> TaskMetadata:
    """Resolve :class:`TaskMetadata` for a function object.

    Args:
        func: Python function or C builtin function object.

    Returns:
        Resolved task metadata.

    Raises:
        ValueError: If a local project function lacks ``@meta(...)`` metadata
            or its metadata has no stable task id.
    """
    if isinstance(func, FunctionType) and is_lambda_function(func):
        return TaskMetadata(lambda_task_id(func))

    if is_local_project_function(func):
        if not hasattr(func, "__task_metadata__"):
            msg = f"Local function {func.__module__}.{func.__qualname__} must define __task_metadata__. Use @meta(...)."
            raise ValueError(msg)
        metadata = cast("TaskMetadata", func.__task_metadata__)
        if not metadata.id:
            name = f"{func.__module__}.{func.__qualname__}"
            msg = f"Local function {name} has no task id. Set @meta(id=...) or run `misen fill`."
            raise ValueError(msg)
        return metadata

    return TaskMetadata(external_callable_id(func))


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
