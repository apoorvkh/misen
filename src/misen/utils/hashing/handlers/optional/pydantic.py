"""Handlers for pydantic models."""

import importlib.util
from typing import Any

from misen.utils.hashing.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry

__all__ = ["pydantic_handlers", "pydantic_handlers_by_type"]


pydantic_handlers: HandlerTypeList = []
pydantic_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("pydantic") is not None:

    def _is_pydantic_model(obj: Any) -> bool:
        return any(
            base.__name__ == "BaseModel" and base.__module__.split(".")[0] == "pydantic"
            for base in type(obj).__mro__
        )

    class PydanticModelHandler(CollectionHandler):
        """Hash pydantic models by field name/value pairs (stable across pydantic versions)."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _is_pydantic_model(obj)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            # Use direct field access rather than model_dump() for version stability.
            fields = obj.model_fields if hasattr(obj, "model_fields") else obj.__fields__
            return [(name, getattr(obj, name)) for name in fields]

    pydantic_handlers = [PydanticModelHandler]
    pydantic_handlers_by_type = {"pydantic.main.BaseModel": PydanticModelHandler}
