"""Serializers for polars DataFrame and Series."""

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.utils.serde.base import Serializer

__all__ = ["polars_serializers", "polars_serializers_by_type"]

polars_serializers: list[type[Serializer]] = []
polars_serializers_by_type: dict[str, type[Serializer]] = {}

if importlib.util.find_spec("polars") is not None:

    class PolarsDataFrameSerializer(Serializer[Any]):
        """Serialize ``polars.DataFrame`` via Parquet."""

        @staticmethod
        def match(obj: Any) -> bool:
            import polars as pl

            return isinstance(obj, pl.DataFrame)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import polars as pl

            obj.write_parquet(directory / "data.parquet")
            return {"polars_version": pl.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import polars as pl

            return pl.read_parquet(directory / "data.parquet")

    class PolarsSeriesSerializer(Serializer[Any]):
        """Serialize ``polars.Series`` via Parquet (single-column DataFrame)."""

        @staticmethod
        def match(obj: Any) -> bool:
            import polars as pl

            return isinstance(obj, pl.Series)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import polars as pl

            df = obj.to_frame()
            df.write_parquet(directory / "data.parquet")
            return {
                "polars_version": pl.__version__,
                "series_name": obj.name,
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            import polars as pl

            df = pl.read_parquet(directory / "data.parquet")
            series = df.to_series(0)
            series_name = meta.get("series_name")
            if series_name is not None:
                series = series.alias(series_name)
            return series

    polars_serializers = [PolarsDataFrameSerializer, PolarsSeriesSerializer]
    polars_serializers_by_type = {
        "polars.dataframe.frame.DataFrame": PolarsDataFrameSerializer,
        "polars.series.series.Series": PolarsSeriesSerializer,
    }
