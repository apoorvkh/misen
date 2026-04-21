"""Serializer for plotly figures via JSON."""

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.utils.serde.base import Serializer

__all__ = ["plotly_serializers", "plotly_serializers_by_type"]

plotly_serializers: list[type[Serializer]] = []
plotly_serializers_by_type: dict[str, type[Serializer]] = {}

if importlib.util.find_spec("plotly") is not None:

    class PlotlyFigureSerializer(Serializer[Any]):
        """Serialize ``plotly.graph_objects.Figure`` via JSON (full round-trip)."""

        @staticmethod
        def match(obj: Any) -> bool:
            from plotly.basedatatypes import BaseFigure

            return isinstance(obj, BaseFigure)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import plotly

            json_str = obj.to_json()
            (directory / "figure.json").write_text(json_str, encoding="utf-8")
            return {"plotly_version": plotly.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import plotly.io as pio

            json_str = (directory / "figure.json").read_text(encoding="utf-8")
            return pio.from_json(json_str)

    plotly_serializers = [PlotlyFigureSerializer]
    plotly_serializers_by_type = {"plotly.graph_objs._figure.Figure": PlotlyFigureSerializer}
