"""Serializer for python-docx Document via .docx format."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["docx_serializers", "docx_serializers_by_type"]

docx_serializers: SerializerTypeList = []
docx_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("docx") is not None:
    from pathlib import Path

    class DocxDocumentSerializer(Serializer[Any]):
        """Serialize ``python-docx`` ``Document`` via ``.docx`` format."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            from docx import Document
            from docx.document import Document as DocumentClass

            _ = Document  # ensure import works
            return isinstance(obj, DocumentClass)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            obj.save(directory / "document.docx")
            write_meta(directory, DocxDocumentSerializer)

        @staticmethod
        def load(directory: Path) -> Any:
            from docx import Document

            return Document(directory / "document.docx")

    docx_serializers = [DocxDocumentSerializer]
    docx_serializers_by_type = {"docx.document.Document": DocxDocumentSerializer}
