"""Handlers for OpenCV keypoint/match/UMat objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import hash_msgspec, incremental_hash

__all__ = ["opencv_handlers", "opencv_handlers_by_type"]


opencv_handlers: HandlerTypeList = []
opencv_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("cv2") is not None:
    _numpy_available = importlib.util.find_spec("numpy") is not None

    class OpenCVKeyPointHandler(PrimitiveHandler):
        """Hash cv2.KeyPoint objects by geometry and score metadata."""

        @staticmethod
        def match(obj: Any) -> bool:
            return type(obj).__module__ == "cv2" and type(obj).__qualname__ == "KeyPoint"

        @staticmethod
        def digest(obj: Any) -> int:
            return hash_msgspec(
                (
                    tuple(float(v) for v in obj.pt),
                    float(obj.size),
                    float(obj.angle),
                    float(obj.response),
                    int(obj.octave),
                    int(obj.class_id),
                )
            )

    class OpenCVDMatchHandler(PrimitiveHandler):
        """Hash cv2.DMatch objects by index/distance metadata."""

        @staticmethod
        def match(obj: Any) -> bool:
            return type(obj).__module__ == "cv2" and type(obj).__qualname__ == "DMatch"

        @staticmethod
        def digest(obj: Any) -> int:
            return hash_msgspec((int(obj.queryIdx), int(obj.trainIdx), int(obj.imgIdx), float(obj.distance)))

    class OpenCVUMatHandler(CollectionHandler):
        """Hash cv2.UMat objects by dtype/shape/content."""

        @staticmethod
        def match(obj: Any) -> bool:
            return type(obj).__module__ == "cv2" and type(obj).__qualname__ == "UMat"

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            if not _numpy_available:
                msg = "numpy must be installed if using cv2 objects in misen. Please `pip install numpy`."
                raise ImportError(msg)

            import numpy as np

            array = np.ascontiguousarray(obj.get())
            payload_hash = incremental_hash(lambda sink: sink.write(array.tobytes()))
            return [str(array.dtype), tuple(int(dim) for dim in array.shape), payload_hash]

    opencv_handlers = [
        OpenCVKeyPointHandler,
        OpenCVDMatchHandler,
        OpenCVUMatHandler,
    ]
    opencv_handlers_by_type = {
        "cv2.KeyPoint": OpenCVKeyPointHandler,
        "cv2.DMatch": OpenCVDMatchHandler,
        "cv2.UMat": OpenCVUMatHandler,
    }
