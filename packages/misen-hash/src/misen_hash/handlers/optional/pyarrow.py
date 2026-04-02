"""Handlers for declarative pyarrow schema objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
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


    pyarrow_handlers = [PyArrowSchemaHandler]
    pyarrow_handlers_by_type = {
        "pyarrow.lib.Schema": PyArrowSchemaHandler,
    }
