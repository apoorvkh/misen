"""Handlers for scikit-image transform objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry

__all__ = ["skimage_handlers", "skimage_handlers_by_type"]


skimage_handlers: HandlerTypeList = []
skimage_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("skimage") is not None:

    def _is_skimage_transform(obj: Any) -> bool:
        if type(obj).__module__.split(".")[0] != "skimage":
            return False
        return any("Transform" in base.__name__ for base in type(obj).__mro__)

    class SkimageTransformHandler(CollectionHandler):
        """Hash skimage transform objects by class, params matrix, and attributes."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _is_skimage_transform(obj)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            payload = dict(vars(obj))
            return [
                getattr(obj, "params", None),
                payload,
            ]

    skimage_handlers = [SkimageTransformHandler]
    skimage_handlers_by_type = {
        "skimage.transform._geometric.GeometricTransform": SkimageTransformHandler,
    }
