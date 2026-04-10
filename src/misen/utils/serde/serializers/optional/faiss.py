"""Serializer for FAISS indices."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["faiss_serializers", "faiss_serializers_by_type"]

faiss_serializers: SerializerTypeList = []
faiss_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("faiss") is not None:
    from pathlib import Path

    class FaissIndexSerializer(Serializer[Any]):
        """Serialize FAISS ``Index`` via ``write_index``/``read_index``."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import faiss

            return isinstance(obj, faiss.Index)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import faiss

            faiss.write_index(obj, str(directory / "index.faiss"))
            write_meta(directory, FaissIndexSerializer)

        @staticmethod
        def load(directory: Path) -> Any:
            import faiss

            return faiss.read_index(str(directory / "index.faiss"))

    faiss_serializers = [FaissIndexSerializer]
    faiss_serializers_by_type = {}  # FAISS index types vary, rely on match()
