"""Handlers for pandas objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry

__all__ = ["pandas_handlers", "pandas_handlers_by_type"]


pandas_handlers: HandlerTypeList = []
pandas_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("pandas") is not None:

    def _to_uint64_hashes(obj: Any) -> tuple[int, ...]:
        from pandas.util import hash_pandas_object

        hashed = hash_pandas_object(obj, index=True, categorize=False).to_numpy(dtype="uint64", copy=False)
        return tuple(int(value) for value in hashed)

    class PandasIndexHandler(CollectionHandler):
        """Hash pandas Index objects with pandas' stable hasher with fallback."""

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "pandas":
                return False

            import pandas as pd

            return isinstance(obj, pd.Index)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            name = getattr(obj, "name", None)
            dtype = str(getattr(obj, "dtype", ""))
            try:
                values: Any = _to_uint64_hashes(obj)
            except (NotImplementedError, TypeError, ValueError):
                values = list(obj.tolist())
            return [name, dtype, values]


    class PandasSeriesHandler(CollectionHandler):
        """Hash pandas Series with pandas' stable hasher with fallback."""

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "pandas":
                return False

            import pandas as pd

            return isinstance(obj, pd.Series)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            name = obj.name
            dtype = str(obj.dtype)
            try:
                values: Any = _to_uint64_hashes(obj)
            except (NotImplementedError, TypeError, ValueError):
                values = {
                    "name": obj.name,
                    "dtype": str(obj.dtype),
                    "index": list(obj.index.tolist()),
                    "values": list(obj.tolist()),
                }
            return [name, dtype, values]


    class PandasDataFrameHandler(CollectionHandler):
        """Hash pandas DataFrames with pandas' stable hasher with fallback."""

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "pandas":
                return False

            import pandas as pd

            return isinstance(obj, pd.DataFrame)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            columns = list(obj.columns.tolist())
            dtypes = [str(dtype) for dtype in obj.dtypes.tolist()]

            try:
                values: Any = _to_uint64_hashes(obj)
            except (NotImplementedError, TypeError, ValueError):
                values = obj.to_dict(orient="split")
            return [columns, dtypes, values]

    pandas_handlers = [
        PandasIndexHandler,
        PandasSeriesHandler,
        PandasDataFrameHandler,
    ]
    pandas_handlers_by_type = {
        "pandas.core.indexes.base.Index": PandasIndexHandler,
        "pandas.core.series.Series": PandasSeriesHandler,
        "pandas.core.frame.DataFrame": PandasDataFrameHandler,
    }
