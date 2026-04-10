"""Serializer for Keras models via the native ``.keras`` format."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["keras_serializers", "keras_serializers_by_type"]

keras_serializers: SerializerTypeList = []
keras_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("keras") is not None:
    from pathlib import Path

    class KerasModelSerializer(Serializer[Any]):
        """Serialize Keras ``Model`` via the native ``.keras`` format."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import keras

            return isinstance(obj, keras.Model)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import keras

            obj.save(directory / "model.keras")
            write_meta(directory, KerasModelSerializer, keras_version=keras.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            import keras

            return keras.models.load_model(directory / "model.keras")

    keras_serializers = [KerasModelSerializer]
    keras_serializers_by_type = {"keras.src.models.model.Model": KerasModelSerializer}
