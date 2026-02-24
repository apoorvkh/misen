"""Handlers for rustworkx graph objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry

__all__ = ["rustworkx_handlers", "rustworkx_handlers_by_type"]


rustworkx_handlers: HandlerTypeList = []
rustworkx_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("rustworkx") is not None:

    def _sort_key(value: Any) -> tuple[str, str, str]:
        return (type(value).__module__, type(value).__qualname__, repr(value))

    class RustworkxGraphHandler(CollectionHandler):
        """Hash rustworkx PyGraph/PyDiGraph objects by normalized nodes and edges."""

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "rustworkx":
                return False

            import rustworkx as rx

            return isinstance(obj, (rx.PyGraph, rx.PyDiGraph))

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            import rustworkx as rx

            node_entries = sorted(
                ((int(index), obj[index]) for index in obj.node_indices()),
                key=lambda item: (_sort_key(item[1]), item[0]),
            )
            canonical_indices = {index: position for position, (index, _) in enumerate(node_entries)}
            canonical_nodes = [payload for _, payload in node_entries]

            directed = isinstance(obj, rx.PyDiGraph)
            multigraph = bool(getattr(obj, "multigraph", False))

            edge_entries: list[tuple[int, int, Any]] = []
            for source, target, weight in obj.weighted_edge_list():
                canonical_source = canonical_indices[int(source)]
                canonical_target = canonical_indices[int(target)]

                if not directed and canonical_source > canonical_target:
                    canonical_source, canonical_target = canonical_target, canonical_source

                edge_entries.append((canonical_source, canonical_target, weight))

            edge_entries.sort(key=lambda item: (item[0], item[1], _sort_key(item[2])))

            return [directed, multigraph, canonical_nodes, edge_entries]

    rustworkx_handlers = [RustworkxGraphHandler]
    rustworkx_handlers_by_type = {
        "rustworkx.PyGraph": RustworkxGraphHandler,
        "rustworkx.PyDiGraph": RustworkxGraphHandler,
    }
