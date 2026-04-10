"""Serializer for HuggingFace tokenizers (the standalone library)."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["tokenizers_serializers", "tokenizers_serializers_by_type"]

tokenizers_serializers: SerializerTypeList = []
tokenizers_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("tokenizers") is not None:
    from pathlib import Path

    class TokenizersSerializer(Serializer[Any]):
        """Serialize ``tokenizers.Tokenizer`` via JSON string (full round-trip)."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            from tokenizers import Tokenizer

            return isinstance(obj, Tokenizer)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import tokenizers

            json_str = obj.to_str()
            (directory / "tokenizer.json").write_text(json_str, encoding="utf-8")
            write_meta(directory, TokenizersSerializer, tokenizers_version=tokenizers.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            from tokenizers import Tokenizer

            json_str = (directory / "tokenizer.json").read_text(encoding="utf-8")
            return Tokenizer.from_str(json_str)

    tokenizers_serializers = [TokenizersSerializer]
    tokenizers_serializers_by_type = {"tokenizers.Tokenizer": TokenizersSerializer}
