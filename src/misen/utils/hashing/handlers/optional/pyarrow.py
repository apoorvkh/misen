"""Handlers for declarative pyarrow data type objects."""

import importlib.util
from typing import Any

from misen.utils.hashing.handler_base import HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler, hash_values

__all__ = ["pyarrow_handlers", "pyarrow_handlers_by_type"]


pyarrow_handlers: HandlerTypeList = []
pyarrow_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("pyarrow") is not None:

    class PyArrowDataTypeHandler(PrimitiveHandler):
        """Hash pyarrow DataType objects (e.g. pa.int64(), pa.list_(pa.float32()))."""

        @staticmethod
        def match(obj: Any) -> bool:
            import pyarrow as pa

            return isinstance(obj, pa.DataType)

        @staticmethod
        def digest(obj: Any) -> int:
            return hash_values(str(obj))

    pyarrow_handlers = [PyArrowDataTypeHandler]
    pyarrow_handlers_by_type = {
        "pyarrow.lib.DataType": PyArrowDataTypeHandler,
    }
