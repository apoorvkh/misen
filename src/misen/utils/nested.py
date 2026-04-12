"""Generic tree-traversal utilities for nested Python containers.

These functions walk dicts, lists, tuples, sets, and frozensets recursively,
applying a mapper or yielding scalar leaves.  They are intentionally
domain-agnostic and know nothing about tasks, hashing, or workspaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = [
    "iter_nested_leaves",
    "map_nested_leaves",
]


def map_nested_leaves(value: Any, leaf_mapper: Callable[[Any], Any]) -> Any:
    """Map scalar leaves recursively while preserving container structure.

    Args:
        value: Arbitrary nested structure.
        leaf_mapper: Function applied to non-container leaves.

    Returns:
        Structure mirroring ``value`` with mapped leaves.
    """
    value_type = type(value)
    if value_type is dict:
        return {map_nested_leaves(k, leaf_mapper): map_nested_leaves(v, leaf_mapper) for k, v in value.items()}
    if value_type is list:
        return [map_nested_leaves(v, leaf_mapper) for v in value]
    if value_type is tuple:
        return tuple(map_nested_leaves(v, leaf_mapper) for v in value)
    if value_type is set:
        return {map_nested_leaves(v, leaf_mapper) for v in value}
    if value_type is frozenset:
        return frozenset(map_nested_leaves(v, leaf_mapper) for v in value)
    return leaf_mapper(value)


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

    if type(value) in (list, tuple, set, frozenset):
        for nested in value:
            yield from iter_nested_leaves(nested)
        return

    yield value
