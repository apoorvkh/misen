"""Handlers for declarative pyarrow data type objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import canonical_hash

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
            return canonical_hash(str(obj))

    pyarrow_handlers = [PyArrowDataTypeHandler]
    pyarrow_handlers_by_type = {
        "pyarrow.lib.DataType": PyArrowDataTypeHandler,
    }
