"""Handlers for declarative Hugging Face transformers config objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry

__all__ = ["transformers_handlers", "transformers_handlers_by_type"]


transformers_handlers: HandlerTypeList = []
transformers_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("transformers") is not None:

    def _has_transformers_base(obj: Any, type_name: str) -> bool:
        return any(
            base.__name__ == type_name and base.__module__.split(".")[0] == "transformers"
            for base in type(obj).__mro__
        )

    class TransformersConfigHandler(CollectionHandler):
        """Hash transformers configs by class identity and serialized payload."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_transformers_base(obj, "PretrainedConfig")

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            payload = obj.to_dict() if hasattr(obj, "to_dict") else vars(obj)
            return [payload]

    transformers_handlers = [TransformersConfigHandler]
    transformers_handlers_by_type = {
        "transformers.configuration_utils.PretrainedConfig": TransformersConfigHandler,
    }
