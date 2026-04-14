"""Serializer for HuggingFace tokenizers (the standalone library)."""

import importlib.util
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import (
    Serializer,
    SerializerTypeRegistry,
)

__all__ = ["tokenizers_serializers", "tokenizers_serializers_by_type"]

tokenizers_serializers: list[type[Serializer]] = []
tokenizers_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("tokenizers") is not None:
    from pathlib import Path

    class TokenizersSerializer(Serializer[Any]):
        """Serialize ``tokenizers.Tokenizer`` via JSON string (full round-trip)."""

        @staticmethod
        def match(obj: Any) -> bool:
            from tokenizers import Tokenizer

            return isinstance(obj, Tokenizer)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import tokenizers

            json_str = obj.to_str()
            (directory / "tokenizer.json").write_text(json_str, encoding="utf-8")
            return {"tokenizers_version": tokenizers.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            from tokenizers import Tokenizer

            json_str = (directory / "tokenizer.json").read_text(encoding="utf-8")
            return Tokenizer.from_str(json_str)

    tokenizers_serializers = [TokenizersSerializer]
    tokenizers_serializers_by_type = {"tokenizers.Tokenizer": TokenizersSerializer}
