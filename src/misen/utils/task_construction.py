"""Helpers for building task dependency graphs."""

from __future__ import annotations

import itertools
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, cast

from misen.utils.graph import DependencyGraph
from misen.utils.hashes import ResultHash, TaskHash
from misen.utils.nested_args import iter_nested_leaves, map_nested_leaves
from misen.utils.sentinels import ASSIGNED_RESOURCES, WORK_DIR

if TYPE_CHECKING:
    from collections.abc import Mapping
    from inspect import Signature

    from misen.task import Task
    from misen.utils.task_properties import TaskProperties
    from misen.workspace import Workspace


def hash_task_arguments(
    *,
    signature: Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    properties: TaskProperties,
    hash_task_by_result: bool = False,
    workspace: Workspace | Literal["auto"] = "auto",
) -> dict[str, tuple[TaskHash | ResultHash, int]]:
    """Return canonical hashes for bound task arguments."""
    from misen.task import Task

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
    """Collect task dependencies nested in args/kwargs values."""
    from misen.task import Task

    values = itertools.chain(args, kwargs.values())
    leaves = itertools.chain.from_iterable(map(iter_nested_leaves, values))
    return frozenset(leaf for leaf in leaves if isinstance(leaf, Task))
