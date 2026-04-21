"""Serializer for FAISS indices."""

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.utils.serde.base import Serializer

__all__ = ["faiss_serializers", "faiss_serializers_by_type"]

faiss_serializers: list[type[Serializer]] = []
faiss_serializers_by_type: dict[str, type[Serializer]] = {}

if importlib.util.find_spec("faiss") is not None:

    class FaissIndexSerializer(Serializer[Any]):
        """Serialize FAISS ``Index`` via ``write_index``/``read_index``."""

        @staticmethod
        def match(obj: Any) -> bool:
            import faiss

            return isinstance(obj, faiss.Index)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any] | None:
            import faiss

            faiss.write_index(obj, str(directory / "index.faiss"))
            return None

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import faiss

            return faiss.read_index(str(directory / "index.faiss"))

    faiss_serializers = [FaissIndexSerializer]
    faiss_serializers_by_type = {}  # FAISS index types vary, rely on match()
