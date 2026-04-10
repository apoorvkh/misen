"""Serializer for python-pptx Presentation via .pptx format."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["pptx_serializers", "pptx_serializers_by_type"]

pptx_serializers: SerializerTypeList = []
pptx_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("pptx") is not None:
    from pathlib import Path

    class PptxPresentationSerializer(Serializer[Any]):
        """Serialize ``python-pptx`` ``Presentation`` via ``.pptx`` format."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            from pptx import Presentation
            from pptx.presentation import Presentation as PresentationClass

            _ = Presentation  # ensure import works
            return isinstance(obj, PresentationClass)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            obj.save(directory / "presentation.pptx")
            write_meta(directory, PptxPresentationSerializer)

        @staticmethod
        def load(directory: Path) -> Any:
            from pptx import Presentation

            return Presentation(directory / "presentation.pptx")

    pptx_serializers = [PptxPresentationSerializer]
    pptx_serializers_by_type = {"pptx.presentation.Presentation": PptxPresentationSerializer}
