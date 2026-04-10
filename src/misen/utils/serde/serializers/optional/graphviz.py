"""Serializer for Graphviz graph objects via DOT source text."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    qualified_type_name,
    write_meta,
)

__all__ = ["graphviz_serializers", "graphviz_serializers_by_type"]

graphviz_serializers: SerializerTypeList = []
graphviz_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("graphviz") is not None:
    from pathlib import Path

    class GraphvizSerializer(Serializer[Any]):
        """Serialize graphviz ``Graph``/``Digraph``/``Source`` via DOT source text."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import graphviz

            return isinstance(obj, (graphviz.Graph, graphviz.Digraph, graphviz.Source))

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import graphviz

            (directory / "graph.gv").write_text(obj.source, encoding="utf-8")
            write_meta(
                directory,
                GraphvizSerializer,
                graphviz_version=graphviz.__version__,
                graph_type=qualified_type_name(type(obj)),
            )

        @staticmethod
        def load(directory: Path) -> Any:
            import graphviz

            from misen.utils.serde.serializer_base import read_meta

            source = (directory / "graph.gv").read_text(encoding="utf-8")
            meta = read_meta(directory)
            graph_type = meta.get("graph_type", "") if meta else ""

            if "Digraph" in graph_type:
                return graphviz.Source(source)
            if "Graph" in graph_type:
                return graphviz.Source(source)
            return graphviz.Source(source)

    graphviz_serializers = [GraphvizSerializer]
    graphviz_serializers_by_type = {
        "graphviz.graphs.Graph": GraphvizSerializer,
        "graphviz.graphs.Digraph": GraphvizSerializer,
        "graphviz.sources.Source": GraphvizSerializer,
    }
