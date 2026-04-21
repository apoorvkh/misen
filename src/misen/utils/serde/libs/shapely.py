"""Serializer for shapely geometries via Well-Known Binary (WKB)."""

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.utils.serde.base import Serializer

__all__ = ["shapely_serializers", "shapely_serializers_by_type"]

shapely_serializers: list[type[Serializer]] = []
shapely_serializers_by_type: dict[str, type[Serializer]] = {}

if importlib.util.find_spec("shapely") is not None:

    class ShapelyGeometrySerializer(Serializer[Any]):
        """Serialize shapely geometries via WKB (OGC standard, compact, lossless)."""

        @staticmethod
        def match(obj: Any) -> bool:
            import shapely

            return isinstance(obj, shapely.Geometry)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import shapely

            wkb_data = shapely.to_wkb(obj)
            (directory / "data.wkb").write_bytes(wkb_data)
            return {"shapely_version": shapely.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import shapely

            wkb_data = (directory / "data.wkb").read_bytes()
            return shapely.from_wkb(wkb_data)

    shapely_serializers = [ShapelyGeometrySerializer]
    shapely_serializers_by_type = {"shapely.lib.Geometry": ShapelyGeometrySerializer}
