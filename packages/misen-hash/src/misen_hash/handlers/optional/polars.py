"""Handlers for polars objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import hash_msgspec, incremental_hash

__all__ = ["polars_handlers", "polars_handlers_by_type"]


polars_handlers: HandlerTypeList = []
polars_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("polars") is not None:

    def _has_polars_base(obj: Any, type_name: str) -> bool:
        return any(
            base.__name__ == type_name and base.__module__.split(".")[0] == "polars"
            for base in type(obj).__mro__
        )

    class PolarsSeriesHandler(CollectionHandler):
        """Hash polars Series with name/dtype/values."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_polars_base(obj, "Series")

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            return [obj.name, str(obj.dtype), obj.to_list()]

    class PolarsDataFrameHandler(CollectionHandler):
        """Hash polars DataFrames with schema plus row payload."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_polars_base(obj, "DataFrame")

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            schema = [(name, str(dtype)) for name, dtype in obj.schema.items()]
            payload = obj.to_dict(as_series=False)
            return [schema, payload]

    class PolarsLazyFrameHandler(PrimitiveHandler):
        """Hash polars LazyFrame plans."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_polars_base(obj, "LazyFrame")

        @staticmethod
        def digest(obj: Any) -> int:
            if hasattr(obj, "serialize"):
                serialized_hash = incremental_hash(lambda sink: sink.write(obj.serialize()))
                return hash_msgspec(("serialized", serialized_hash))
            return hash_msgspec(("explain_optimized", obj.explain(optimized=True)))

    polars_handlers = [
        PolarsSeriesHandler,
        PolarsDataFrameHandler,
        PolarsLazyFrameHandler,
    ]
    polars_handlers_by_type = {
        "polars.series.series.Series": PolarsSeriesHandler,
        "polars.dataframe.frame.DataFrame": PolarsDataFrameHandler,
        "polars.lazyframe.frame.LazyFrame": PolarsLazyFrameHandler,
    }
