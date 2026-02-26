"""Handlers for Pillow image objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import hash_msgspec, incremental_hash

__all__ = ["pillow_handlers", "pillow_handlers_by_type"]

pillow_handlers: HandlerTypeList = []
pillow_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("PIL") is not None:

    class PillowImageHandler(PrimitiveHandler):
        """Hash Pillow image pixel payload plus mode/shape metadata."""

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "PIL":
                return False

            from PIL import Image

            return isinstance(obj, Image.Image)

        @staticmethod
        def digest(obj: Any) -> int:
            payload_hash = incremental_hash(lambda sink: sink.write(obj.tobytes()))
            return hash_msgspec(
                (
                    obj.mode,
                    tuple(int(dim) for dim in obj.size),
                    payload_hash,
                )
            )

    pillow_handlers = [PillowImageHandler]
    pillow_handlers_by_type = {"PIL.Image.Image": PillowImageHandler}
