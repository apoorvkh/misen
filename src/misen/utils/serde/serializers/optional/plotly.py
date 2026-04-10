"""Serializer for plotly figures via JSON."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["plotly_serializers", "plotly_serializers_by_type"]

plotly_serializers: SerializerTypeList = []
plotly_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("plotly") is not None:
    from pathlib import Path

    class PlotlyFigureSerializer(Serializer[Any]):
        """Serialize ``plotly.graph_objects.Figure`` via JSON (full round-trip)."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            from plotly.basedatatypes import BaseFigure

            return isinstance(obj, BaseFigure)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import plotly

            json_str = obj.to_json()
            (directory / "figure.json").write_text(json_str, encoding="utf-8")
            write_meta(directory, PlotlyFigureSerializer, plotly_version=plotly.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            import plotly.io as pio

            json_str = (directory / "figure.json").read_text(encoding="utf-8")
            return pio.from_json(json_str)

    plotly_serializers = [PlotlyFigureSerializer]
    plotly_serializers_by_type = {"plotly.graph_objs._figure.Figure": PlotlyFigureSerializer}
