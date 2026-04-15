"""Serializers for TensorFlow tensors and sparse tensors."""

import importlib.util
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import (
    Serializer,
    SerializerTypeRegistry,
)

__all__ = ["tensorflow_serializers", "tensorflow_serializers_by_type"]

tensorflow_serializers: list[type[Serializer]] = []
tensorflow_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("tensorflow") is not None and importlib.util.find_spec("numpy") is not None:
    from pathlib import Path

    class TensorFlowTensorSerializer(Serializer[Any]):
        """Serialize ``tf.Tensor`` by converting to numpy and saving as ``.npy``."""

        @staticmethod
        def match(obj: Any) -> bool:
            import tensorflow as tf

            return isinstance(obj, tf.Tensor)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import numpy as np
            import tensorflow as tf

            np.save(directory / "data.npy", obj.numpy(), allow_pickle=False)
            return {
                "tensorflow_version": tf.__version__,
                "dtype": str(obj.dtype.name),
                "shape": list(obj.shape),
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import numpy as np
            import tensorflow as tf

            arr = np.load(directory / "data.npy", allow_pickle=False)
            return tf.constant(arr)

    class TensorFlowSparseTensorSerializer(Serializer[Any]):
        """Serialize ``tf.SparseTensor`` via indices, values, and shape as ``.npy`` files."""

        @staticmethod
        def match(obj: Any) -> bool:
            import tensorflow as tf

            return isinstance(obj, tf.SparseTensor)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import numpy as np
            import tensorflow as tf

            np.save(directory / "indices.npy", obj.indices.numpy(), allow_pickle=False)
            np.save(directory / "values.npy", obj.values.numpy(), allow_pickle=False)
            return {
                "tensorflow_version": tf.__version__,
                "dense_shape": list(obj.dense_shape.numpy()),
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            import numpy as np
            import tensorflow as tf

            indices = np.load(directory / "indices.npy", allow_pickle=False)
            values = np.load(directory / "values.npy", allow_pickle=False)
            return tf.SparseTensor(indices=indices, values=values, dense_shape=meta["dense_shape"])

    tensorflow_serializers = [TensorFlowSparseTensorSerializer, TensorFlowTensorSerializer]
    tensorflow_serializers_by_type = {
        "tensorflow.python.framework.ops.EagerTensor": TensorFlowTensorSerializer,
        "tensorflow.python.framework.sparse_tensor.SparseTensor": TensorFlowSparseTensorSerializer,
    }
