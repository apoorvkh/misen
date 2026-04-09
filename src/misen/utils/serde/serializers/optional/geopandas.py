"""Serializers for geopandas GeoDataFrame and GeoSeries via GeoParquet."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["geopandas_serializers", "geopandas_serializers_by_type"]

geopandas_serializers: SerializerTypeList = []
geopandas_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("geopandas") is not None:
    from pathlib import Path

    class GeopandasDataFrameSerializer(Serializer[Any]):
        """Serialize ``geopandas.GeoDataFrame`` via GeoParquet."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import geopandas as gpd

            return isinstance(obj, gpd.GeoDataFrame)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import geopandas as gpd

            obj.to_parquet(directory / "data.parquet")
            write_meta(directory, GeopandasDataFrameSerializer, geopandas_version=gpd.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            import geopandas as gpd

            return gpd.read_parquet(directory / "data.parquet")

    class GeopandasSeriesSerializer(Serializer[Any]):
        """Serialize ``geopandas.GeoSeries`` via GeoParquet (single-column GeoDataFrame)."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import geopandas as gpd

            return isinstance(obj, gpd.GeoSeries)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import geopandas as gpd

            gdf = gpd.GeoDataFrame(geometry=obj)
            gdf.to_parquet(directory / "data.parquet")
            write_meta(
                directory,
                GeopandasSeriesSerializer,
                geopandas_version=gpd.__version__,
                series_name=obj.name,
            )

        @staticmethod
        def load(directory: Path) -> Any:
            import geopandas as gpd

            from misen.utils.serde.serializer_base import read_meta

            gdf = gpd.read_parquet(directory / "data.parquet")
            meta = read_meta(directory)
            series_name = meta.get("series_name") if meta else None
            series = gdf.geometry
            series.name = series_name
            return series

    geopandas_serializers = [GeopandasDataFrameSerializer, GeopandasSeriesSerializer]
    geopandas_serializers_by_type = {
        "geopandas.geodataframe.GeoDataFrame": GeopandasDataFrameSerializer,
        "geopandas.geoseries.GeoSeries": GeopandasSeriesSerializer,
    }
