"""Serializers for geopandas GeoDataFrame and GeoSeries via GeoParquet."""

import importlib.util
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import (
    Serializer,
    SerializerTypeRegistry,
)

__all__ = ["geopandas_serializers", "geopandas_serializers_by_type"]

geopandas_serializers: list[type[Serializer]] = []
geopandas_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("geopandas") is not None:
    from pathlib import Path

    class GeopandasDataFrameSerializer(Serializer[Any]):
        """Serialize ``geopandas.GeoDataFrame`` via GeoParquet."""

        @staticmethod
        def match(obj: Any) -> bool:
            import geopandas as gpd

            return isinstance(obj, gpd.GeoDataFrame)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import geopandas as gpd

            obj.to_parquet(directory / "data.parquet")
            return {"geopandas_version": gpd.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import geopandas as gpd

            return gpd.read_parquet(directory / "data.parquet")

    class GeopandasSeriesSerializer(Serializer[Any]):
        """Serialize ``geopandas.GeoSeries`` via GeoParquet (single-column GeoDataFrame)."""

        @staticmethod
        def match(obj: Any) -> bool:
            import geopandas as gpd

            return isinstance(obj, gpd.GeoSeries)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import geopandas as gpd

            gdf = gpd.GeoDataFrame(geometry=obj)
            gdf.to_parquet(directory / "data.parquet")
            return {
                "geopandas_version": gpd.__version__,
                "series_name": obj.name,
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            import geopandas as gpd

            gdf = gpd.read_parquet(directory / "data.parquet")
            series = gdf.geometry
            series.name = meta.get("series_name")
            return series

    geopandas_serializers = [GeopandasDataFrameSerializer, GeopandasSeriesSerializer]
    geopandas_serializers_by_type = {
        "geopandas.geodataframe.GeoDataFrame": GeopandasDataFrameSerializer,
        "geopandas.geoseries.GeoSeries": GeopandasSeriesSerializer,
    }
