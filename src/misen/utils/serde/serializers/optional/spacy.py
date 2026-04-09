"""Serializer for spaCy Doc objects."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["spacy_serializers", "spacy_serializers_by_type"]

spacy_serializers: SerializerTypeList = []
spacy_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("spacy") is not None:
    from pathlib import Path

    class SpacyDocSerializer(Serializer[Any]):
        """Serialize spaCy ``Doc`` objects via ``to_bytes``/``from_bytes``.

        Both the Doc and its Vocab are serialized so loading does not require
        the original pipeline model to be installed.
        """

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            from spacy.tokens import Doc

            return isinstance(obj, Doc)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import spacy

            (directory / "vocab.bin").write_bytes(obj.vocab.to_bytes())
            (directory / "doc.bin").write_bytes(obj.to_bytes())
            write_meta(directory, SpacyDocSerializer, spacy_version=spacy.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            from spacy.tokens import Doc
            from spacy.vocab import Vocab

            vocab = Vocab()
            vocab.from_bytes((directory / "vocab.bin").read_bytes())
            return Doc(vocab).from_bytes((directory / "doc.bin").read_bytes())

    spacy_serializers = [SpacyDocSerializer]
    spacy_serializers_by_type = {"spacy.tokens.doc.Doc": SpacyDocSerializer}
