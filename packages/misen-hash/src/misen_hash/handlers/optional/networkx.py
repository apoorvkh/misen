"""Handlers for networkx graph objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry

__all__ = ["networkx_handlers", "networkx_handlers_by_type"]


networkx_handlers: HandlerTypeList = []
networkx_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("networkx") is not None:

    def _sort_key(value: Any) -> tuple[str, str, str]:
        return (type(value).__module__, type(value).__qualname__, repr(value))

    class NetworkXGraphHandler(CollectionHandler):
        """Hash networkx graph topology and attributes."""

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "networkx":
                return False

            import networkx as nx

            return isinstance(obj, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            node_entries = sorted(
                ((node, dict(attrs)) for node, attrs in obj.nodes(data=True)),
                key=lambda item: _sort_key(item[0]),
            )

            directed = bool(obj.is_directed())
            multigraph = bool(obj.is_multigraph())

            if multigraph:
                edge_entries = sorted(
                    ((u, v, key, dict(attrs)) for u, v, key, attrs in obj.edges(keys=True, data=True)),
                    key=lambda item: (_sort_key(item[0]), _sort_key(item[1]), _sort_key(item[2])),
                )
            else:
                edge_entries = sorted(
                    ((u, v, dict(attrs)) for u, v, attrs in obj.edges(data=True)),
                    key=lambda item: (_sort_key(item[0]), _sort_key(item[1])),
                )

            return [
                directed,
                multigraph,
                dict(obj.graph),
                node_entries,
                edge_entries,
            ]

    networkx_handlers = [NetworkXGraphHandler]
    networkx_handlers_by_type = {
        "networkx.classes.graph.Graph": NetworkXGraphHandler,
        "networkx.classes.digraph.DiGraph": NetworkXGraphHandler,
        "networkx.classes.multigraph.MultiGraph": NetworkXGraphHandler,
        "networkx.classes.multidigraph.MultiDiGraph": NetworkXGraphHandler,
    }
