"""Handlers for Hugging Face tokenizers objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import hash_msgspec

__all__ = ["tokenizers_handlers", "tokenizers_handlers_by_type"]


tokenizers_handlers: HandlerTypeList = []
tokenizers_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("tokenizers") is not None:

    class HFTokenizerHandler(PrimitiveHandler):
        """Hash a tokenizers.Tokenizer by its JSON definition."""

        @staticmethod
        def match(obj: Any) -> bool:
            return type(obj).__module__.split(".")[0] == "tokenizers" and hasattr(obj, "to_str")

        @staticmethod
        def digest(obj: Any) -> int:
            return hash_msgspec(obj.to_str())

    tokenizers_handlers = [HFTokenizerHandler]
    tokenizers_handlers_by_type = {"tokenizers.Tokenizer": HFTokenizerHandler}
