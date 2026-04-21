"""Serializer for Pillow (PIL) images."""

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.utils.serde.base import Serializer

__all__ = ["pillow_serializers", "pillow_serializers_by_type"]

pillow_serializers: list[type[Serializer]] = []
pillow_serializers_by_type: dict[str, type[Serializer]] = {}

if importlib.util.find_spec("PIL") is not None:

    class PillowImageSerializer(Serializer[Any]):
        """Serialize ``PIL.Image.Image`` as lossless PNG."""

        @staticmethod
        def match(obj: Any) -> bool:
            from PIL import Image

            return isinstance(obj, Image.Image)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            obj.save(directory / "data.png", format="PNG")
            return {"mode": obj.mode, "size": list(obj.size)}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            from PIL import Image

            return Image.open(directory / "data.png").copy()

    pillow_serializers = [PillowImageSerializer]
    pillow_serializers_by_type = {"PIL.Image.Image": PillowImageSerializer}
