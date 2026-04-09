"""Handlers for msgspec types."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry

__all__ = ["msgspec_handlers", "msgspec_handlers_by_type"]

msgspec_handlers: HandlerTypeList = []
msgspec_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("msgspec") is not None:

    class MsgspecStructHandler(CollectionHandler):
        """Hash msgspec Struct instances by their declared fields."""

        @staticmethod
        def match(obj: Any) -> bool:
            import msgspec

            return isinstance(obj, msgspec.Struct)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            return [(field, getattr(obj, field)) for field in obj.__struct_fields__]

    msgspec_handlers = [MsgspecStructHandler]
    msgspec_handlers_by_type = {"msgspec.Struct": MsgspecStructHandler}
