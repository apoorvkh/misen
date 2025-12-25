from __future__ import annotations

from typing import Generic, TypeVar

import rustworkx as rx

T = TypeVar("T")


class DependencyGraph(rx.PyDiGraph, Generic[T]):
    def copy(self) -> DependencyGraph[T]:
        new = type(self)()
        old_indices = self.node_indices()
        new_indices = new.add_nodes_from([self[i] for i in old_indices])
        remap = dict(zip(old_indices, new_indices))
        edge_list = [(remap[u], remap[v], w) for (u, v, w) in self.weighted_edge_list()]
        new.extend_from_weighted_edge_list(edge_list)
        return new

    def evaluation_order(self) -> list[int]:
        return list(rx.topological_sort(self))[::-1]

    def coarsen_to_anchors(self, anchors: list[int]) -> DependencyGraph[T]:
        graph = self.copy()
        for node in reversed(graph.node_indices()):
            if node not in anchors:
                graph.remove_node_retain_edges(node)
        return graph

    def pretty_print(
        self,
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
        for node_index in self.node_indices():
            node = self[node_index]
            deps = list(self.successors(node_index))
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
