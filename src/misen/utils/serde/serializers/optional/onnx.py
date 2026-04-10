"""Serializer for ONNX models via protobuf."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["onnx_serializers", "onnx_serializers_by_type"]

onnx_serializers: SerializerTypeList = []
onnx_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("onnx") is not None:
    from pathlib import Path

    class ONNXModelSerializer(Serializer[Any]):
        """Serialize ``onnx.ModelProto`` via protobuf binary format."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import onnx

            return isinstance(obj, onnx.ModelProto)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import onnx

            onnx.save_model(obj, str(directory / "model.onnx"))
            write_meta(directory, ONNXModelSerializer, onnx_version=onnx.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            import onnx

            return onnx.load_model(str(directory / "model.onnx"))

    onnx_serializers = [ONNXModelSerializer]
    onnx_serializers_by_type = {"onnx.onnx_ml_pb2.ModelProto": ONNXModelSerializer}
