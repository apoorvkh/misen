from __future__ import annotations

from operator import eq
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import rustworkx as rx

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["DependencyGraph"]

T = TypeVar("T")


class DependencyGraph(Generic[T]):
    def __init__(self) -> None:
        self._g = rx.PyDiGraph(check_cycle=True, multigraph=False)

    ## Wrapper functions

    def copy(self) -> DependencyGraph[T]:
        new = DependencyGraph()
        new._g = self._g.copy()
        return new

    def nodes(self) -> list[T]:
        return self._g.nodes()

    def node_indices(self) -> list[int]:
        return list(self._g.node_indices())

    def add_node(self, node: T) -> int:
        return self._g.add_node(node)

    def __getitem__(self, key: int) -> T:
        return self._g.__getitem__(key)

    def __setitem__(self, key: int, value: T) -> None:
        self._g.__setitem__(key, value)

    def add_edge(self, parent: int, child: int, edge: Any = None) -> None:
        self._g.add_edge(parent, child, edge)

    def successors(self, node_index: int) -> list[T]:
        return self._g.successors(node_index)

    def is_root(self, node_index: int) -> bool:
        return self._g.in_degree(node_index) == 0

    def remove_node_by_value(self, value: Any, cmp: Callable[[Any, Any], bool] = eq, first: bool = False) -> None:
        for node_index in self._g.node_indices():
            if cmp(self._g[node_index], value):
                self._g.remove_node(node_index)
                if first:
                    break

    ## Custom functions

    def evaluation_order(self) -> list[int]:
        return list(rx.topological_sort(self._g))[::-1]

    def coarsen_to_anchors(self, anchors: list[int]) -> None:
        for node in reversed(self._g.node_indices()):
            if node not in anchors:
                self._g.remove_node_retain_edges(node)

    def pretty_print(
        self,
        *,
        roots: list[T] | None = None,
        max_depth: int | None = None,
        show_duplicates: bool = False,
    ) -> None:
        """
        Pretty-print a dependency graph as a hierarchy.

        Interprets edges as: u -> v  means "u depends on v" (so v is printed under u).
        """

        def sort_key(x: T) -> str:
            # Deterministic ordering even for unorderable node types
            return str(x)

        # Collect all nodes + dependencies
        all_nodes: list[T] = []
        all_deps: set[T] = set()
        adjacency: dict[T, list[T]] = {}
        for node_index in self._g.node_indices():
            node = self[node_index]
            deps = list(self._g.successors(node_index))
            adjacency[node] = deps
            all_nodes.append(node)
            all_deps.update(deps)

        if roots is None:
            # Roots are nodes that are not a dependency of any other node (no incoming edges)
            roots = [n for n in all_nodes if n not in all_deps]
            roots.sort(key=sort_key)

        printed: set[T] = set()

        def walk(node: T, prefix: str, is_last: bool, depth: int, stack: set[T]) -> None:
            connector = "└── " if is_last else "├── "

            if node in stack:
                print(prefix + connector + f"{node} (cycle)")
                return

            if (not show_duplicates) and (node in printed):
                print(prefix + connector + f"{node} (↩︎)")
                return

            print(prefix + connector + str(node))
            printed.add(node)

            if max_depth is not None and depth >= max_depth:
                return

            children = list(adjacency.get(node, ()))
            children.sort(key=sort_key)

            new_prefix = prefix + ("    " if is_last else "│   ")
            stack2 = set(stack)
            stack2.add(node)

            for i, child in enumerate(children):
                walk(child, new_prefix, i == len(children) - 1, depth + 1, stack2)

        for r_i, root in enumerate(roots):
            # Print root without a connector for a cleaner look
            if r_i:
                print()  # blank line between root trees
            print(str(root))

            children = list(adjacency.get(root, ()))
            children.sort(key=sort_key)
            for i, child in enumerate(children):
                walk(child, "", i == len(children) - 1, 1, {root})
