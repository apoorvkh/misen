"""Serializer for Keras models via the native ``.keras`` format."""

import importlib.util
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import (
    Serializer,
    SerializerTypeRegistry,
)

__all__ = ["keras_serializers", "keras_serializers_by_type"]

keras_serializers: list[type[Serializer]] = []
keras_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("keras") is not None:
    from pathlib import Path

    class KerasModelSerializer(Serializer[Any]):
        """Serialize Keras ``Model`` via the native ``.keras`` format."""

        @staticmethod
        def match(obj: Any) -> bool:
            import keras

            return isinstance(obj, keras.Model)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import keras

            obj.save(directory / "model.keras")
            return {"keras_version": keras.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import keras

            return keras.models.load_model(directory / "model.keras")

    keras_serializers = [KerasModelSerializer]
    keras_serializers_by_type = {"keras.src.models.model.Model": KerasModelSerializer}
