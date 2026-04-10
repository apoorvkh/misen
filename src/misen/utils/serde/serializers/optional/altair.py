"""Serializer for Altair charts via JSON."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["altair_serializers", "altair_serializers_by_type"]

altair_serializers: SerializerTypeList = []
altair_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("altair") is not None:
    from pathlib import Path

    class AltairChartSerializer(Serializer[Any]):
        """Serialize Altair charts via Vega-Lite JSON (full round-trip)."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import altair as alt

            return isinstance(obj, alt.TopLevelMixin)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import altair as alt

            json_str = obj.to_json()
            (directory / "chart.json").write_text(json_str, encoding="utf-8")
            write_meta(directory, AltairChartSerializer, altair_version=alt.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            import altair as alt

            json_str = (directory / "chart.json").read_text(encoding="utf-8")
            return alt.Chart.from_json(json_str)

    altair_serializers = [AltairChartSerializer]
    altair_serializers_by_type = {
        "altair.vegalite.v5.api.Chart": AltairChartSerializer,
        "altair.vegalite.v5.api.LayerChart": AltairChartSerializer,
        "altair.vegalite.v5.api.HConcatChart": AltairChartSerializer,
        "altair.vegalite.v5.api.VConcatChart": AltairChartSerializer,
        "altair.vegalite.v5.api.FacetChart": AltairChartSerializer,
        "altair.vegalite.v5.api.ConcatChart": AltairChartSerializer,
        "altair.vegalite.v5.api.RepeatChart": AltairChartSerializer,
    }
