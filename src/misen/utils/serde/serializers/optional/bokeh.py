"""Serializer for Bokeh documents via JSON."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["bokeh_serializers", "bokeh_serializers_by_type"]

bokeh_serializers: SerializerTypeList = []
bokeh_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("bokeh") is not None:
    from pathlib import Path

    class BokehFigureSerializer(Serializer[Any]):
        """Serialize Bokeh figures/plots via JSON."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            from bokeh.model import Model

            return isinstance(obj, Model)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import bokeh
            from bokeh.embed import json_item

            json_data = json_item(obj)
            import json

            (directory / "figure.json").write_text(json.dumps(json_data), encoding="utf-8")
            write_meta(directory, BokehFigureSerializer, bokeh_version=bokeh.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            import json

            from bokeh.io import from_json

            json_str = (directory / "figure.json").read_text(encoding="utf-8")
            data = json.loads(json_str)
            return from_json(json.dumps(data.get("doc", data)))

    bokeh_serializers = [BokehFigureSerializer]
    bokeh_serializers_by_type = {}  # Bokeh model types vary, rely on match()
