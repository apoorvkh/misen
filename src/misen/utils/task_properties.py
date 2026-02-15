"""Helpers for @task decorator normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar

from msgspec import Struct

from misen.utils.functions import (
    external_callable_id,
    is_lambda_function,
    is_local_project_function,
    lambda_task_id,
)
from misen.utils.hashes import ResultHash
from misen.utils.object_io import DefaultSerializer, Serializer

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FunctionType


__all__ = ["Resources", "TaskProperties", "task"]

P = ParamSpec("P")
R = TypeVar("R")


class TaskProperties(Struct, frozen=True):
    """Immutable metadata describing how a Task should be identified and cached."""

    id: str
    cache: bool = False
    exclude: set[str] = set()
    defaults: dict[str, Any] = {}
    versions: dict[tuple[str, ResultHash], int] = {}
    index_by: Literal["task", "result"] = "result"
    resources: Callable[..., Resources] = lambda *_, **__: Resources()
    serializer: type[Serializer] = DefaultSerializer


class Resources(Struct, frozen=True):
    """Resource requirements for executing a Task."""

    time: int | None = None
    nodes: int = 1
    memory: int = 8
    cpus: int = 1
    gpus: int = 0
    gpu_memory: int | None = None


def task(
    *,
    id: str | None = None,  # noqa: A002
    cache: bool = False,
    exclude: set[str] | None = None,
    defaults: dict[str, Any] | None = None,
    versions: dict[str, dict[Any, int]] | None = None,
    index_by: Literal["task", "result"] = "result",
    resources: Callable[..., Resources] | Resources | None = None,
    serializer: type[Serializer[R]] = DefaultSerializer,  # TODO: typing
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to control how a Task is identified and cached.

    Attaches `__task_properties__: TaskProperties` attribute to `self.func`.

    Arguments:
        id:
            Stable identifier for the task definition. Will raise ValueError if None.
        cache:
            If True, `Task.result()` may store results in the Workspace.
        exclude:
            Exclude arguments (by name) from hashing.
        defaults:
            If an argument value matches the provided default, it is omitted from hashing.
        versions:
            For versioning per (argument, value) pair. Normalized to a {argument : ResultHash(value)} mapping.
        index_by:
            Determines how result is indexed (i.e. how ResultHash is computed) in Workspace:
            - "task": index by resolved task hash
            - "result": index by the result object
        resources:
            Optional resource resolver for each Task instance.
            - If a callable: called with the task arguments during `Task(...)` construction.
            - If Resources: used directly for every `Task(...)`.
        serializer:
            Serializer type for saving/loading results.

    Returns:
        A decorator that mutates `func` by setting `func.__task_properties__`.
    """
    if id is None:
        msg = "id must be provided."
        raise ValueError(msg)

    if resources is None:
        resources = Resources()
    if isinstance(resources, Resources):
        resources = lambda r=resources, *_, **__: r  # noqa: E731

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        """Attach task properties to the decorated function."""
        func.__task_properties__ = TaskProperties(  # ty:ignore[unresolved-attribute]
            id=id,
            cache=cache,
            exclude=(exclude or set()),
            defaults=(defaults or {}),
            versions=_normalize_versions(versions),
            index_by=index_by,
            resources=resources,
            serializer=serializer,
        )

        return func

    return decorator


def resolve_task_properties(func: FunctionType) -> TaskProperties:
    """Return TaskProperties for a function object."""
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
    """Normalize argument version mapping into hash-keyed lookup form."""
    return {
        (name, ResultHash.from_object(value)): version
        for name, value_to_version in (versions or {}).items()
        for value, version in value_to_version.items()
    }
