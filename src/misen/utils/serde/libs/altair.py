"""Serializer for Altair charts via JSON."""

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.utils.serde.base import Serializer

__all__ = ["altair_serializers", "altair_serializers_by_type"]

altair_serializers: list[type[Serializer]] = []
altair_serializers_by_type: dict[str, type[Serializer]] = {}

if importlib.util.find_spec("altair") is not None:

    class AltairChartSerializer(Serializer[Any]):
        """Serialize Altair charts via Vega-Lite JSON (full round-trip)."""

        @staticmethod
        def match(obj: Any) -> bool:
            import altair as alt

            return isinstance(obj, alt.TopLevelMixin)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import altair as alt

            json_str = obj.to_json()
            (directory / "chart.json").write_text(json_str, encoding="utf-8")
            return {"altair_version": alt.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
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
