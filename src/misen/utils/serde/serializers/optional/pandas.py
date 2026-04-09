"""Serializers for pandas DataFrame and Series."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["pandas_serializers", "pandas_serializers_by_type"]

pandas_serializers: SerializerTypeList = []
pandas_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("pandas") is not None:
    from pathlib import Path

    class PandasDataFrameSerializer(Serializer[Any]):
        """Serialize ``pandas.DataFrame`` via Parquet."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import pandas as pd

            return isinstance(obj, pd.DataFrame)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import pandas as pd

            obj.to_parquet(directory / "data.parquet")
            write_meta(directory, PandasDataFrameSerializer, pandas_version=pd.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            import pandas as pd

            return pd.read_parquet(directory / "data.parquet")

    class PandasSeriesSerializer(Serializer[Any]):
        """Serialize ``pandas.Series`` via Parquet (single-column DataFrame)."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import pandas as pd

            return isinstance(obj, pd.Series)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import pandas as pd

            df = obj.to_frame(name="__series__")
            df.to_parquet(directory / "data.parquet")
            write_meta(
                directory,
                PandasSeriesSerializer,
                pandas_version=pd.__version__,
                series_name=obj.name,
            )

        @staticmethod
        def load(directory: Path) -> Any:
            import pandas as pd

            from misen.utils.serde.serializer_base import read_meta

            df = pd.read_parquet(directory / "data.parquet")
            meta = read_meta(directory)
            series_name = meta.get("series_name") if meta else None
            series = df["__series__"]
            series.name = series_name
            return series

    pandas_serializers = [PandasDataFrameSerializer, PandasSeriesSerializer]
    pandas_serializers_by_type = {
        "pandas.core.frame.DataFrame": PandasDataFrameSerializer,
        "pandas.core.series.Series": PandasSeriesSerializer,
    }
