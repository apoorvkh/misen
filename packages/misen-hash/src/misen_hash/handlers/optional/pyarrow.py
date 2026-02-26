"""Handlers for pyarrow objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import hash_msgspec

__all__ = ["pyarrow_handlers", "pyarrow_handlers_by_type"]


pyarrow_handlers: HandlerTypeList = []
pyarrow_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("pyarrow") is not None:

    def _has_pyarrow_base(obj: Any, type_name: str) -> bool:
        return any(
            base.__name__ == type_name and base.__module__.split(".")[0] == "pyarrow"
            for base in type(obj).__mro__
        )

    def _schema_payload(schema: Any) -> str:
        if hasattr(schema, "to_string"):
            return schema.to_string(show_schema_metadata=True)
        return str(schema)

    class PyArrowSchemaHandler(PrimitiveHandler):
        """Hash pyarrow Schema objects."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_pyarrow_base(obj, "Schema")

        @staticmethod
        def digest(obj: Any) -> int:
            return hash_msgspec(_schema_payload(obj))


    class PyArrowScalarHandler(CollectionHandler):
        """Hash pyarrow Scalar objects."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_pyarrow_base(obj, "Scalar")

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            return [str(obj.type), obj.as_py()]


    class PyArrowArrayHandler(CollectionHandler):
        """Hash pyarrow Array objects."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_pyarrow_base(obj, "Array")

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            return [str(obj.type), obj.to_pylist()]


    class PyArrowChunkedArrayHandler(CollectionHandler):
        """Hash pyarrow ChunkedArray objects."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_pyarrow_base(obj, "ChunkedArray")

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            return [str(obj.type), [chunk.to_pylist() for chunk in obj.chunks]]


    class PyArrowRecordBatchHandler(CollectionHandler):
        """Hash pyarrow RecordBatch objects."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_pyarrow_base(obj, "RecordBatch")

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            return [_schema_payload(obj.schema), obj.to_pydict()]


    class PyArrowTableHandler(CollectionHandler):
        """Hash pyarrow Table objects."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_pyarrow_base(obj, "Table")

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            return [_schema_payload(obj.schema), obj.to_pydict()]

    pyarrow_handlers = [
        PyArrowSchemaHandler,
        PyArrowScalarHandler,
        PyArrowArrayHandler,
        PyArrowChunkedArrayHandler,
        PyArrowRecordBatchHandler,
        PyArrowTableHandler,
    ]
    pyarrow_handlers_by_type = {
        "pyarrow.lib.Schema": PyArrowSchemaHandler,
        "pyarrow.lib.Scalar": PyArrowScalarHandler,
        "pyarrow.lib.Array": PyArrowArrayHandler,
        "pyarrow.lib.ChunkedArray": PyArrowChunkedArrayHandler,
        "pyarrow.lib.RecordBatch": PyArrowRecordBatchHandler,
        "pyarrow.lib.Table": PyArrowTableHandler,
    }
