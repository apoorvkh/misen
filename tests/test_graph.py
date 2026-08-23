"""Dependency graph behavior."""
# ruff: noqa: D103, S101

from io import StringIO
from operator import is_

import pytest

from misen.utils.graph import DependencyGraph


def test_evaluation_order_respects_dependencies() -> None:
    graph: DependencyGraph[str] = DependencyGraph()
    root, left, right, leaf, unrelated = [graph.add_node(name) for name in ("root", "left", "right", "leaf", "x")]
    graph.add_edge(root, left)
    graph.add_edge(root, right)
    graph.add_edge(left, leaf)
    graph.add_edge(right, leaf)

    order = graph.evaluation_order()
    positions = {node: position for position, node in enumerate(order)}
    assert positions[leaf] < positions[left] < positions[root]
    assert positions[leaf] < positions[right] < positions[root]
    assert unrelated in order
    assert list(graph) == [graph[index] for index in order]


def test_edges_are_unique_and_cycles_are_detected() -> None:
    graph: DependencyGraph[str] = DependencyGraph()
    first, second, third = [graph.add_node(name) for name in ("first", "second", "third")]
    graph.add_edge(first, second)
    graph.add_edge(first, second)
    graph.add_edge(third, first)  # A valid backward-index edge.

    assert graph.successors(first) == ["second"]
    graph.add_edge(first, third)
    with pytest.raises(ValueError, match="contains a cycle"):
        graph.evaluation_order()

    self_cycle: DependencyGraph[str] = DependencyGraph()
    node = self_cycle.add_node("node")
    self_cycle.add_edge(node, node)
    with pytest.raises(ValueError, match="contains a cycle"):
        self_cycle.evaluation_order()
    with pytest.raises(ValueError, match="contains a cycle"):
        self_cycle.coarsen_to_anchors([])


def test_successors_and_roots_use_dependency_semantics() -> None:
    graph: DependencyGraph[str] = DependencyGraph()
    parent, first, second = [graph.add_node(name) for name in ("parent", "first", "second")]
    graph.add_edge(parent, first)
    graph.add_edge(parent, second)

    assert graph.successors(parent) == ["second", "first"]
    assert graph.is_root(parent)
    assert not graph.is_root(first)
    assert not graph.is_root(second)


def test_copy_has_independent_values_and_topology() -> None:
    graph: DependencyGraph[str] = DependencyGraph()
    root = graph.add_node("root")
    leaf = graph.add_node("leaf")
    graph.add_edge(root, leaf)

    copied = graph.copy()
    copied[root] = "changed"
    copied.remove_node_by_value("leaf")

    assert graph.nodes() == ["root", "leaf"]
    assert graph.successors(root) == ["leaf"]
    assert copied.nodes() == ["changed"]
    assert copied.node_indices() == [root]


def test_remove_node_by_value_preserves_stable_indices() -> None:
    shared = object()
    graph: DependencyGraph[object] = DependencyGraph()
    first = graph.add_node(shared)
    middle = graph.add_node(object())
    last = graph.add_node(shared)

    graph.remove_node_by_value(shared, cmp=is_, first=True)
    assert graph.node_indices() == [middle, last]
    with pytest.raises(IndexError, match="No node"):
        _ = graph[first]

    graph.remove_node_by_value(shared, cmp=is_)
    assert graph.node_indices() == [middle]


def test_removing_a_node_does_not_retain_edges() -> None:
    graph: DependencyGraph[str] = DependencyGraph()
    root, middle, leaf = [graph.add_node(name) for name in ("root", "middle", "leaf")]
    graph.add_edge(root, middle)
    graph.add_edge(middle, leaf)

    graph.remove_node_by_value("middle")

    assert graph.successors(root) == []
    assert graph.is_root(leaf)


def test_coarsening_retains_and_deduplicates_anchor_edges() -> None:
    graph: DependencyGraph[str] = DependencyGraph()
    root, left, right, leaf = [graph.add_node(name) for name in ("root", "left", "right", "leaf")]
    graph.add_edge(root, left)
    graph.add_edge(root, right)
    graph.add_edge(left, leaf)
    graph.add_edge(right, leaf)

    graph.coarsen_to_anchors([root, leaf])

    assert graph.node_indices() == [root, leaf]
    assert graph.successors(root) == ["leaf"]
    assert graph.evaluation_order() == [leaf, root]


def test_long_chain_uses_iterative_traversal() -> None:
    size = 5_000
    graph: DependencyGraph[int] = DependencyGraph()
    for value in range(size):
        graph.add_node(value)
    for value in range(size - 1):
        graph.add_edge(value + 1, value)

    order = graph.evaluation_order()
    assert len(order) == size
    assert order[0] == 0
    assert order[-1] == size - 1


def test_pretty_print() -> None:
    graph: DependencyGraph[str] = DependencyGraph()
    root = graph.add_node("root")
    leaf = graph.add_node("leaf")
    graph.add_edge(root, leaf)
    target = StringIO()

    graph.pretty_print(target=target)

    assert target.getvalue() == "root\n└── leaf\n"
