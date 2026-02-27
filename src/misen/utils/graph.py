"""Dependency graph utilities built on top of ``rustworkx``.

Edge convention used across misen: ``A -> B`` means "A depends on B".
Evaluation order is therefore reverse topological order.
"""

from __future__ import annotations

import sys
from operator import eq
from typing import TYPE_CHECKING, Any, Generic, TextIO, TypeVar

import rustworkx as rx

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = ["DependencyGraph"]

T = TypeVar("T")


class DependencyGraph(Generic[T]):
    """Directed-acyclic graph wrapper with dependency semantics."""

    __slots__ = ("_g",)

    def __init__(self) -> None:
        """Initialize an empty dependency graph."""
        self._g = rx.PyDiGraph(check_cycle=True, multigraph=False)

    ## Wrapper functions

    def copy(self) -> DependencyGraph[T]:
        """Return a shallow copy of the graph."""
        new = DependencyGraph()
        new._g = self._g.copy()
        return new

    def nodes(self) -> list[T]:
        """Return node values in storage order."""
        return self._g.nodes()

    def node_indices(self) -> list[int]:
        """Return node indices in storage order."""
        return list(self._g.node_indices())

    def add_node(self, node: T) -> int:
        """Add a node and return its index."""
        return self._g.add_node(node)

    def __getitem__(self, key: int) -> T:
        """Return the node value at the given index."""
        return self._g.__getitem__(key)

    def __setitem__(self, key: int, value: T) -> None:
        """Replace the node value at the given index."""
        self._g.__setitem__(key, value)

    def add_edge(self, parent: int, child: int, edge: Any = None) -> None:
        """Add an edge from parent to child with optional edge data."""
        self._g.add_edge(parent, child, edge)

    def successors(self, node_index: int) -> list[T]:
        """Return successors of the given node index."""
        return self._g.successors(node_index)

    def is_root(self, node_index: int) -> bool:
        """Return True when the node has no incoming edges."""
        return self._g.in_degree(node_index) == 0

    def remove_node_by_value(self, value: Any, *, cmp: Callable[[Any, Any], bool] = eq, first: bool = False) -> None:
        """Remove nodes that compare equal to the given value.

        Args:
            value: The value to match against nodes.
            cmp: Comparator for matching nodes against the value.
            first: Whether to remove only the first matching node.
        """
        indices_to_remove = []
        for node_index in self._g.node_indices():
            if cmp(self._g[node_index], value):
                indices_to_remove.append(node_index)
                if first:
                    break
        self._g.remove_nodes_from(indices_to_remove)

    ## Custom functions

    def evaluation_order(self) -> list[int]:
        """Return node indices in dependency evaluation order.

        Returns:
            Node indices ordered so dependencies appear before dependents.
        """
        return list(rx.topological_sort(self._g))[::-1]

    def __iter__(self) -> Iterator[T]:
        """Yield node values in dependency evaluation order."""
        for i in self.evaluation_order():
            yield self._g[i]

    def coarsen_to_anchors(self, anchors: list[int]) -> None:
        """Remove non-anchor nodes while retaining induced anchor edges.

        Args:
            anchors: Node indices to keep.
        """
        for node in reversed(self._g.node_indices()):
            if node not in anchors:
                self._g.remove_node_retain_edges(node)

    def pretty_print(
        self,
        *,
        roots: list[T] | None = None,
        max_depth: int | None = None,
        show_duplicates: bool = False,
        target: TextIO | None = None,
    ) -> None:
        """Pretty-print a dependency graph as a hierarchy.

        Interprets edges as: u -> v  means "u depends on v" (so v is printed under u).

        Args:
            roots: Optional list of root nodes to start from.
            max_depth: Optional maximum depth to render.
            show_duplicates: If True, show repeated nodes instead of back-references.
            target: Stream to write to (defaults to sys.stdout).
        """
        stream = sys.stdout if target is None else target

        def write_line(text: str = "") -> None:
            """Write a line to the target stream."""
            stream.write(f"{text}\n")

        def sort_key(node: T) -> str:
            """Return deterministic sort key for pretty-print ordering."""
            # Deterministic ordering even for unorderable node types.
            return str(node)

        all_nodes: list[T] = []
        all_dependencies: set[T] = set()
        adjacency: dict[T, list[T]] = {}
        for node_index in self._g.node_indices():
            node = self[node_index]
            dependencies = list(self._g.successors(node_index))
            adjacency[node] = dependencies
            all_nodes.append(node)
            all_dependencies.update(dependencies)

        if roots is None:
            # Roots are nodes that are not a dependency of any other node (no incoming edges)
            roots = [node for node in all_nodes if node not in all_dependencies]
            roots.sort(key=sort_key)

        printed: set[T] = set()

        def walk(node: T, prefix: str, *, is_last: bool, depth: int, stack: set[T]) -> None:
            """Recursively print a node subtree with indentation."""
            connector = "└── " if is_last else "├── "

            if node in stack:
                write_line(prefix + connector + f"{node} (cycle)")
                return

            if (not show_duplicates) and (node in printed):
                write_line(prefix + connector + f"{node} (↩︎)")
                return

            write_line(prefix + connector + str(node))
            printed.add(node)

            if max_depth is not None and depth >= max_depth:
                return

            children = list(adjacency.get(node, ()))
            children.sort(key=sort_key)

            new_prefix = prefix + ("    " if is_last else "│   ")
            stack2 = set(stack)
            stack2.add(node)

            for i, child in enumerate(children):
                walk(child, new_prefix, is_last=i == len(children) - 1, depth=depth + 1, stack=stack2)

        for r_i, root in enumerate(roots):
            # Print root without a connector for a cleaner look
            if r_i:
                write_line()  # blank line between root trees
            write_line(str(root))

            children = list(adjacency.get(root, ()))
            children.sort(key=sort_key)
            for i, child in enumerate(children):
                walk(child, "", is_last=i == len(children) - 1, depth=1, stack={root})
