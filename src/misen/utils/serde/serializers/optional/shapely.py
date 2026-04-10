"""Serializer for shapely geometries via Well-Known Binary (WKB)."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["shapely_serializers", "shapely_serializers_by_type"]

shapely_serializers: SerializerTypeList = []
shapely_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("shapely") is not None:
    from pathlib import Path

    class ShapelyGeometrySerializer(Serializer[Any]):
        """Serialize shapely geometries via WKB (OGC standard, compact, lossless)."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import shapely

            return isinstance(obj, shapely.Geometry)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import shapely

            wkb_data = shapely.to_wkb(obj)
            (directory / "data.wkb").write_bytes(wkb_data)
            write_meta(directory, ShapelyGeometrySerializer, shapely_version=shapely.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            import shapely

            wkb_data = (directory / "data.wkb").read_bytes()
            return shapely.from_wkb(wkb_data)

    shapely_serializers = [ShapelyGeometrySerializer]
    shapely_serializers_by_type = {"shapely.lib.Geometry": ShapelyGeometrySerializer}
