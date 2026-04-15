"""Handlers for attrs classes."""

import importlib.util
from typing import Any

from misen.utils.hashing.base import CollectionHandler, Handler, HandlerTypeRegistry

__all__ = ["attrs_handlers", "attrs_handlers_by_type"]

attrs_handlers: list[type[Handler]] = []
attrs_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("attrs") is not None:

    class AttrsHandler(CollectionHandler):
        """Hash attrs instances by field/value pairs."""

        @staticmethod
        def match(obj: Any) -> bool:
            return hasattr(type(obj), "__attrs_attrs__")

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            import attrs

            return [(field.name, getattr(obj, field.name)) for field in attrs.fields(type(obj))]

    attrs_handlers = [AttrsHandler]
    attrs_handlers_by_type = {}
