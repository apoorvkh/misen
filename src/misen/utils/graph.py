"""Directed dependency graph utilities.

Edge convention used across Misen: ``A -> B`` means "A depends on B".
"""

from __future__ import annotations

from operator import eq
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = ["DependencyGraph"]

T = TypeVar("T")


class _Removed:
    """Marker for a removed node slot."""


_REMOVED = _Removed()


class DependencyGraph(Generic[T]):
    """Directed dependency graph with stable node indices."""

    __slots__ = ("_dependencies", "_dependents", "_nodes")

    def __init__(self) -> None:
        """Initialize an empty dependency graph."""
        self._nodes: list[T | _Removed] = []
        self._dependencies: list[dict[int, None]] = []
        self._dependents: list[dict[int, None]] = []

    def _node(self, index: int) -> T:
        """Return an active node, rejecting missing and removed indices."""
        if index < 0 or index >= len(self._nodes) or self._nodes[index] is _REMOVED:
            msg = f"No node at index {index}."
            raise IndexError(msg)
        return cast("T", self._nodes[index])

    def _add_edge(self, parent: int, child: int) -> None:
        """Add an edge to both adjacency indexes."""
        self._dependencies[parent][child] = None
        self._dependents[child][parent] = None

    def _remove_node(self, index: int) -> None:
        """Remove a node and all its incident edges."""
        self._node(index)
        for dependency in self._dependencies[index]:
            self._dependents[dependency].pop(index)
        for dependent in self._dependents[index]:
            self._dependencies[dependent].pop(index)
        self._dependencies[index].clear()
        self._dependents[index].clear()
        self._nodes[index] = _REMOVED

    def nodes(self) -> list[T]:
        """Return node values in storage order."""
        return [cast("T", node) for node in self._nodes if node is not _REMOVED]

    def node_indices(self) -> list[int]:
        """Return active node indices in storage order."""
        return [index for index, node in enumerate(self._nodes) if node is not _REMOVED]

    def add_node(self, node: T) -> int:
        """Add a node and return its index."""
        index = len(self._nodes)
        self._nodes.append(node)
        self._dependencies.append({})
        self._dependents.append({})
        return index

    def __getitem__(self, key: int) -> T:
        """Return the node value at the given index."""
        return self._node(key)

    def __setitem__(self, key: int, value: T) -> None:
        """Replace the node value at the given index."""
        self._node(key)
        self._nodes[key] = value

    def add_edge(self, parent: int, child: int, edge: Any = None) -> None:
        """Add an edge from a dependent node to one of its dependencies.

        Args:
            parent: Index of the dependent node.
            child: Index of the dependency node.
            edge: Ignored edge data, retained for API compatibility.
        """
        del edge
        self._node(parent)
        self._node(child)
        self._add_edge(parent, child)

    def successors(self, node_index: int) -> list[T]:
        """Return the dependency values of the given node."""
        self._node(node_index)
        return [self._node(index) for index in reversed(self._dependencies[node_index])]

    def is_root(self, node_index: int) -> bool:
        """Return whether no other node depends on the given node."""
        self._node(node_index)
        return not self._dependents[node_index]

    def remove_node_by_value(self, value: Any, *, cmp: Callable[[Any, Any], bool] = eq, first: bool = False) -> None:
        """Remove nodes that compare equal to the given value.

        Args:
            value: The value to match against nodes.
            cmp: Comparator for matching nodes against the value.
            first: Whether to remove only the first matching node.
        """
        for node_index in self.node_indices():
            if cmp(self[node_index], value):
                self._remove_node(node_index)
                if first:
                    break

    def evaluation_order(self) -> list[int]:
        """Return indices ordered so dependencies precede dependents.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        active = self.node_indices()
        remaining = [len(dependencies) for dependencies in self._dependencies]
        ready = [index for index in reversed(active) if remaining[index] == 0]
        order: list[int] = []

        while ready:
            node = ready.pop()
            order.append(node)
            for dependent in reversed(self._dependents[node]):
                remaining[dependent] -= 1
                if remaining[dependent] == 0:
                    ready.append(dependent)

        if len(order) != len(active):
            msg = "Dependency graph contains a cycle."
            raise ValueError(msg)
        return order

    def __iter__(self) -> Iterator[T]:
        """Yield node values in dependency evaluation order."""
        for index in self.evaluation_order():
            yield self[index]

    def coarsen_to_anchors(self, anchors: list[int]) -> None:
        """Remove non-anchor nodes while retaining induced anchor edges.

        Args:
            anchors: Node indices to keep.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        self.evaluation_order()
        anchor_set = set(anchors)
        for node in reversed(self.node_indices()):
            if node in anchor_set:
                continue
            for dependent in tuple(self._dependents[node]):
                for dependency in self._dependencies[node]:
                    self._add_edge(dependent, dependency)
            self._remove_node(node)
