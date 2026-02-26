"""Handlers for sentencepiece processor objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import hash_msgspec, incremental_hash

__all__ = ["sentencepiece_handlers", "sentencepiece_handlers_by_type"]


sentencepiece_handlers: HandlerTypeList = []
sentencepiece_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("sentencepiece") is not None:

    class SentencePieceProcessorHandler(PrimitiveHandler):
        """Hash sentencepiece models by serialized model proto."""

        @staticmethod
        def match(obj: Any) -> bool:
            return (
                type(obj).__module__.split(".")[0] == "sentencepiece"
                and type(obj).__qualname__ == "SentencePieceProcessor"
            )

        @staticmethod
        def digest(obj: Any) -> int:
            if hasattr(obj, "serialized_model_proto"):
                payload = bytes(obj.serialized_model_proto())
                payload_hash = incremental_hash(lambda sink: sink.write(payload))
                return hash_msgspec(("serialized_model_proto", payload_hash))

            to_piece = obj.id_to_piece if hasattr(obj, "id_to_piece") else obj.IdToPiece
            pieces = [to_piece(i) for i in range(int(obj.get_piece_size()))]
            return hash_msgspec(("vocabulary", pieces))

    sentencepiece_handlers = [SentencePieceProcessorHandler]
    sentencepiece_handlers_by_type = {"sentencepiece.SentencePieceProcessor": SentencePieceProcessorHandler}
