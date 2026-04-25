"""Serializer for networkx graphs via GraphML.

GraphML is an XML-based interchange format with a published schema,
so saved graphs survive networkx version bumps.  All four graph
types — :class:`~networkx.Graph`, :class:`~networkx.DiGraph`,
:class:`~networkx.MultiGraph`, :class:`~networkx.MultiDiGraph` —
round-trip faithfully *within the GraphML attribute model*:
node/edge identifiers and primitive-typed attribute values are
preserved; the multigraph flag is captured in the on-disk meta.

GraphML stores all identifiers as XML strings.  We record the
original node ID type (and edge-key type for multigraphs) in meta
so :func:`networkx.read_graphml` casts them back on load — ``int``
and ``str`` IDs both survive.

Faithfulness restrictions (raise :class:`~misen.exceptions.SerializationError`
at save time):

- Node IDs must be homogeneously typed and limited to ``int`` /
  ``str``.  Tuples, floats, custom objects, or mixed-type IDs would
  silently degrade on read.
- Attribute values (graph-level, node, and edge) must be ``bool`` /
  ``int`` / ``float`` / ``str`` (the GraphML attribute domain).
  Anything richer (ndarrays, dicts, custom objects) wouldn't survive.
"""

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.exceptions import SerializationError
from misen.utils.serde.base import Serializer

__all__ = ["networkx_serializers", "networkx_serializers_by_type"]

networkx_serializers: list[type[Serializer]] = []
networkx_serializers_by_type: dict[str, type[Serializer]] = {}

# GraphML's attribute schema only knows these scalar types.
_GRAPHML_SCALARS = (bool, int, float, str)
# Identifier types we can faithfully cast back via ``read_graphml(node_type=...)``.
_GRAPHML_ID_TYPES: dict[type, str] = {int: "int", str: "str"}
_GRAPHML_ID_TYPE_BY_NAME: dict[str, type] = {"int": int, "str": str}


def _detect_id_type(values: Any) -> type | None:
    """Return the single ID type used in *values*, or ``None`` if empty."""
    types = {type(v) for v in values}
    if not types:
        return None
    if len(types) > 1:
        msg = (
            f"GraphML requires homogeneously-typed identifiers; got mixed types {sorted(t.__name__ for t in types)}. "
            "Convert all node IDs (or edge keys) to a single str/int type before saving."
        )
        raise SerializationError(msg)
    (id_type,) = types
    if id_type not in _GRAPHML_ID_TYPES:
        msg = (
            f"GraphML supports only int/str identifiers; got {id_type.__name__}. "
            "Convert IDs before saving."
        )
        raise SerializationError(msg)
    return id_type


def _check_attrs_serializable(graph: Any) -> None:
    """Walk graph, edge, and node attributes and reject non-GraphML values."""
    bad: list[str] = []
    for key, val in graph.graph.items():
        if not isinstance(val, _GRAPHML_SCALARS):
            bad.append(f"graph[{key!r}] = {type(val).__name__}")
    for node, data in graph.nodes(data=True):
        for key, val in data.items():
            if not isinstance(val, _GRAPHML_SCALARS):
                bad.append(f"nodes[{node!r}][{key!r}] = {type(val).__name__}")
    for u, v, data in graph.edges(data=True):
        for key, val in data.items():
            if not isinstance(val, _GRAPHML_SCALARS):
                bad.append(f"edges[{u!r},{v!r}][{key!r}] = {type(val).__name__}")
    if bad:
        msg = (
            "GraphML only supports primitive attribute values (bool/int/float/str); "
            "the following attributes need to be normalized before serialization: " + ", ".join(bad[:6])
        )
        if len(bad) > 6:
            msg += f" (+{len(bad) - 6} more)"
        raise SerializationError(msg)


if importlib.util.find_spec("networkx") is not None:

    class NetworkXGraphSerializer(Serializer[Any]):
        """Serialize ``networkx`` graphs (Graph/DiGraph/Multi*Graph) via GraphML.

        Round-trips graph type (directed/multi flags), node and edge
        identifiers (``int`` / ``str``), and primitive-typed attributes.
        Any input that GraphML can't faithfully represent raises
        :class:`SerializationError` at save time.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            import networkx as nx

            return isinstance(obj, nx.Graph)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import networkx as nx

            _check_attrs_serializable(obj)
            node_type = _detect_id_type(obj.nodes())
            edge_key_type: type | None = None
            if obj.is_multigraph():
                # MultiGraph edge iteration yields ``(u, v, key)`` triples; we want
                # just the keys.
                edge_key_type = _detect_id_type(k for _, _, k in obj.edges(keys=True))

            nx.write_graphml(obj, directory / "graph.graphml")
            return {
                "networkx_version": nx.__version__,
                "is_directed": bool(obj.is_directed()),
                "is_multigraph": bool(obj.is_multigraph()),
                # ``None`` for an empty graph — read uses the default ``str``.
                "node_type": _GRAPHML_ID_TYPES[node_type] if node_type else None,
                "edge_key_type": _GRAPHML_ID_TYPES[edge_key_type] if edge_key_type else None,
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            import networkx as nx

            node_type = _GRAPHML_ID_TYPE_BY_NAME.get(meta.get("node_type") or "", str)
            edge_key_type = _GRAPHML_ID_TYPE_BY_NAME.get(meta.get("edge_key_type") or "", int)
            g = nx.read_graphml(
                directory / "graph.graphml",
                node_type=node_type,
                edge_key_type=edge_key_type,
            )
            # ``read_graphml`` picks the directed/undirected base from the
            # GraphML header but doesn't distinguish multi from simple — that
            # flag lives in our subdir meta.  Re-route through the desired
            # concrete class so the type the user saved is the type they read.
            is_multigraph = bool(meta.get("is_multigraph", False))
            is_directed = bool(meta.get("is_directed", False))
            if is_multigraph:
                cls: Any = nx.MultiDiGraph if is_directed else nx.MultiGraph
            else:
                cls = nx.DiGraph if is_directed else nx.Graph
            if type(g) is cls:
                return g
            new = cls()
            new.graph.update(g.graph)
            new.add_nodes_from(g.nodes(data=True))
            if is_multigraph:
                # Preserve edge keys when re-routing to the multi class.
                new.add_edges_from((u, v, k, d) for u, v, k, d in g.edges(keys=True, data=True))
            else:
                new.add_edges_from(g.edges(data=True))
            return new

    networkx_serializers = [NetworkXGraphSerializer]
    networkx_serializers_by_type = {
        # Subclasses dispatch via MRO walk in TypeDispatchRegistry.
        "networkx.classes.graph.Graph": NetworkXGraphSerializer,
    }
