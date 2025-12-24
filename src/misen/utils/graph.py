from typing import TypeVar

import rustworkx as rx

T = TypeVar("T")


class Graph(dict[T, set[T]]):
    def to_rustworkx(self) -> rx.PyDiGraph:
        dag = rx.PyDiGraph(check_cycle=True, multigraph=False)
        nodes: dict[T, int] = {t: dag.add_node(t) for t in self}
        for t in self:
            n = nodes[t]
            for d in self[t]:
                dag.add_edge(n, nodes[d], None)
        return dag

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

        # Collect all nodes (keys + deps)
        all_nodes: set[T] = set(self.keys())
        all_deps: set[T] = set()
        for deps in self.values():
            all_nodes.update(deps)
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

            children = list(self.get(node, ()))
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

            children = list(self.get(root, ()))
            children.sort(key=sort_key)
            for i, child in enumerate(children):
                walk(child, "", i == len(children) - 1, 1, {root})
