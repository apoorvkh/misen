"""Utilities for traversing and mapping builtin nested container arguments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = ["iter_nested_leaves", "map_nested_leaves"]


def iter_nested_leaves(value: Any) -> Iterator[Any]:
    """Yield scalar leaves from supported nested containers."""
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
    """Recursively map scalar leaves while preserving common container structure."""
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
