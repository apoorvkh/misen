"""Serializer for ONNX models via protobuf."""

import importlib.util
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import (
    Serializer,
    SerializerTypeRegistry,
)

__all__ = ["onnx_serializers", "onnx_serializers_by_type"]

onnx_serializers: list[type[Serializer]] = []
onnx_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("onnx") is not None:
    from pathlib import Path

    class ONNXModelSerializer(Serializer[Any]):
        """Serialize ``onnx.ModelProto`` via protobuf binary format."""

        @staticmethod
        def match(obj: Any) -> bool:
            import onnx

            return isinstance(obj, onnx.ModelProto)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import onnx

            onnx.save_model(obj, str(directory / "model.onnx"))
            return {"onnx_version": onnx.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import onnx

            return onnx.load_model(str(directory / "model.onnx"))

    onnx_serializers = [ONNXModelSerializer]
    onnx_serializers_by_type = {"onnx.onnx_ml_pb2.ModelProto": ONNXModelSerializer}
