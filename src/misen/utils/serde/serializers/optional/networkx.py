"""Serializer for networkx graphs via node-link JSON encoded as msgpack."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["networkx_serializers", "networkx_serializers_by_type"]

networkx_serializers: SerializerTypeList = []
networkx_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("networkx") is not None:
    from pathlib import Path

    import msgspec.msgpack

    from misen.utils.serde.serializers.stdlib import _decode_tagged, _encode_tagged, _is_msgpack_safe

    class NetworkXGraphSerializer(Serializer[Any]):
        """Serialize networkx graphs via node-link data encoded as msgpack.

        Node and edge attributes must be msgpack-safe types.
        """

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import networkx as nx

            if not isinstance(obj, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                return False
            try:
                data = nx.node_link_data(obj)
                return _is_msgpack_safe(data)
            except (TypeError, ValueError):
                return False

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import networkx as nx

            data = nx.node_link_data(obj)
            tagged = _encode_tagged(data)
            encoded = msgspec.msgpack.encode(tagged)
            (directory / "data.msgpack").write_bytes(encoded)
            write_meta(directory, NetworkXGraphSerializer, networkx_version=nx.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            import networkx as nx

            raw = msgspec.msgpack.decode((directory / "data.msgpack").read_bytes())
            data = _decode_tagged(raw)
            return nx.node_link_graph(data)

    networkx_serializers = [NetworkXGraphSerializer]
    networkx_serializers_by_type = {
        "networkx.classes.graph.Graph": NetworkXGraphSerializer,
        "networkx.classes.digraph.DiGraph": NetworkXGraphSerializer,
        "networkx.classes.multigraph.MultiGraph": NetworkXGraphSerializer,
        "networkx.classes.multidigraph.MultiDiGraph": NetworkXGraphSerializer,
    }
