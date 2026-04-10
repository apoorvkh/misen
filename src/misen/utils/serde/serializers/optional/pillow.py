"""Serializer for Pillow (PIL) images."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["pillow_serializers", "pillow_serializers_by_type"]

pillow_serializers: SerializerTypeList = []
pillow_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("PIL") is not None:
    from pathlib import Path

    class PillowImageSerializer(Serializer[Any]):
        """Serialize ``PIL.Image.Image`` as lossless PNG."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            from PIL import Image

            return isinstance(obj, Image.Image)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            obj.save(directory / "data.png", format="PNG")
            write_meta(directory, PillowImageSerializer, mode=obj.mode, size=list(obj.size))

        @staticmethod
        def load(directory: Path) -> Any:
            from PIL import Image

            return Image.open(directory / "data.png").copy()

    pillow_serializers = [PillowImageSerializer]
    pillow_serializers_by_type = {"PIL.Image.Image": PillowImageSerializer}
