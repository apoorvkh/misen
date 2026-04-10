"""Serializers for TensorFlow tensors and sparse tensors."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["tensorflow_serializers", "tensorflow_serializers_by_type"]

tensorflow_serializers: SerializerTypeList = []
tensorflow_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("tensorflow") is not None and importlib.util.find_spec("numpy") is not None:
    from pathlib import Path

    class TensorFlowTensorSerializer(Serializer[Any]):
        """Serialize ``tf.Tensor`` by converting to numpy and saving as ``.npy``."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import tensorflow as tf

            return isinstance(obj, tf.Tensor)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import numpy as np
            import tensorflow as tf

            np.save(directory / "data.npy", obj.numpy(), allow_pickle=False)
            write_meta(
                directory,
                TensorFlowTensorSerializer,
                tensorflow_version=tf.__version__,
                dtype=str(obj.dtype.name),
                shape=list(obj.shape),
            )

        @staticmethod
        def load(directory: Path) -> Any:
            import numpy as np
            import tensorflow as tf

            arr = np.load(directory / "data.npy", allow_pickle=False)
            return tf.constant(arr)

    class TensorFlowSparseTensorSerializer(Serializer[Any]):
        """Serialize ``tf.SparseTensor`` via indices, values, and shape as ``.npy`` files."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import tensorflow as tf

            return isinstance(obj, tf.SparseTensor)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import numpy as np
            import tensorflow as tf

            np.save(directory / "indices.npy", obj.indices.numpy(), allow_pickle=False)
            np.save(directory / "values.npy", obj.values.numpy(), allow_pickle=False)
            write_meta(
                directory,
                TensorFlowSparseTensorSerializer,
                tensorflow_version=tf.__version__,
                dense_shape=list(obj.dense_shape.numpy()),
            )

        @staticmethod
        def load(directory: Path) -> Any:
            import numpy as np
            import tensorflow as tf

            from misen.utils.serde.serializer_base import read_meta

            indices = np.load(directory / "indices.npy", allow_pickle=False)
            values = np.load(directory / "values.npy", allow_pickle=False)
            meta = read_meta(directory)
            dense_shape = meta["dense_shape"] if meta else None
            return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    tensorflow_serializers = [TensorFlowSparseTensorSerializer, TensorFlowTensorSerializer]
    tensorflow_serializers_by_type = {
        "tensorflow.python.framework.ops.EagerTensor": TensorFlowTensorSerializer,
        "tensorflow.python.framework.sparse_tensor.SparseTensor": TensorFlowSparseTensorSerializer,
    }
