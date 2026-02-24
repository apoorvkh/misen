"""Handlers for spaCy language pipelines."""

import importlib.util
from typing import Any

from misen_hash.handler_base import HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import hash_msgspec, incremental_hash

__all__ = ["spacy_handlers", "spacy_handlers_by_type"]


spacy_handlers: HandlerTypeList = []
spacy_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("spacy") is not None:

    class SpacyLanguageHandler(PrimitiveHandler):
        """Hash spaCy Language objects by config/meta plus bytes payload."""

        @staticmethod
        def match(obj: Any) -> bool:
            return type(obj).__module__.split(".")[0] == "spacy" and any(
                base.__name__ == "Language" for base in type(obj).__mro__
            )

        @staticmethod
        def digest(obj: Any) -> int:
            payload = bytes(obj.to_bytes())
            payload_hash = incremental_hash(lambda sink: sink.write(payload))
            config_payload = obj.config.to_str() if hasattr(obj.config, "to_str") else str(obj.config)
            return hash_msgspec(
                (
                    obj.meta,
                    config_payload,
                    payload_hash,
                )
            )

    spacy_handlers = [SpacyLanguageHandler]
    spacy_handlers_by_type = {"spacy.language.Language": SpacyLanguageHandler}
