"""Serializers for polars DataFrame and Series."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["polars_serializers", "polars_serializers_by_type"]

polars_serializers: SerializerTypeList = []
polars_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("polars") is not None:
    from pathlib import Path

    class PolarsDataFrameSerializer(Serializer[Any]):
        """Serialize ``polars.DataFrame`` via Parquet."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import polars as pl

            return isinstance(obj, pl.DataFrame)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import polars as pl

            obj.write_parquet(directory / "data.parquet")
            write_meta(directory, PolarsDataFrameSerializer, polars_version=pl.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            import polars as pl

            return pl.read_parquet(directory / "data.parquet")

    class PolarsSeriesSerializer(Serializer[Any]):
        """Serialize ``polars.Series`` via Parquet (single-column DataFrame)."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import polars as pl

            return isinstance(obj, pl.Series)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import polars as pl

            df = obj.to_frame()
            df.write_parquet(directory / "data.parquet")
            write_meta(
                directory,
                PolarsSeriesSerializer,
                polars_version=pl.__version__,
                series_name=obj.name,
            )

        @staticmethod
        def load(directory: Path) -> Any:
            import polars as pl

            from misen.utils.serde.serializer_base import read_meta

            df = pl.read_parquet(directory / "data.parquet")
            meta = read_meta(directory)
            series_name = meta.get("series_name") if meta else None
            series = df.to_series(0)
            if series_name is not None:
                series = series.alias(series_name)
            return series

    polars_serializers = [PolarsDataFrameSerializer, PolarsSeriesSerializer]
    polars_serializers_by_type = {
        "polars.dataframe.frame.DataFrame": PolarsDataFrameSerializer,
        "polars.series.series.Series": PolarsSeriesSerializer,
    }
