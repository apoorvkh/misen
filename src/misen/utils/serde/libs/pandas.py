"""Serializers for pandas DataFrame and Series."""

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.utils.serde.base import Serializer

__all__ = ["pandas_serializers", "pandas_serializers_by_type"]

pandas_serializers: list[type[Serializer]] = []
pandas_serializers_by_type: dict[str, type[Serializer]] = {}

if importlib.util.find_spec("pandas") is not None:

    class PandasDataFrameSerializer(Serializer[Any]):
        """Serialize ``pandas.DataFrame`` via Parquet."""

        @staticmethod
        def match(obj: Any) -> bool:
            import pandas as pd

            return isinstance(obj, pd.DataFrame)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import pandas as pd

            obj.to_parquet(directory / "data.parquet")
            return {"pandas_version": pd.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import pandas as pd

            return pd.read_parquet(directory / "data.parquet")

    class PandasSeriesSerializer(Serializer[Any]):
        """Serialize ``pandas.Series`` via Parquet (single-column DataFrame)."""

        @staticmethod
        def match(obj: Any) -> bool:
            import pandas as pd

            return isinstance(obj, pd.Series)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import pandas as pd

            df = obj.to_frame(name="__series__")
            df.to_parquet(directory / "data.parquet")
            return {
                "pandas_version": pd.__version__,
                "series_name": obj.name,
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            import pandas as pd

            df = pd.read_parquet(directory / "data.parquet")
            series = df["__series__"]
            series.name = meta.get("series_name")
            return series

    pandas_serializers = [PandasDataFrameSerializer, PandasSeriesSerializer]
    pandas_serializers_by_type = {
        "pandas.core.frame.DataFrame": PandasDataFrameSerializer,
        "pandas.core.series.Series": PandasSeriesSerializer,
    }
