"""TensorFlow v2 serializers — batched tensors, per-instance sparse tensors."""

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.utils.serde.base import LeafSerializer, Serializer

__all__ = ["tensorflow_serializers", "tensorflow_serializers_by_type"]

tensorflow_serializers: list[type[Serializer]] = []
tensorflow_serializers_by_type: dict[str, type[Serializer]] = {}

if importlib.util.find_spec("tensorflow") is not None and importlib.util.find_spec("numpy") is not None:

    class TensorFlowTensorSerializer(LeafSerializer[Any]):
        """Batched leaf for ``tf.Tensor`` — one ``tensors.npz`` per save."""

        leaf_kind = "tf_tensor"

        @staticmethod
        def match(obj: Any) -> bool:
            import tensorflow as tf

            return isinstance(obj, tf.Tensor)

        @classmethod
        def to_payload(cls, obj: Any) -> Any:
            return obj.numpy()

        @staticmethod
        def write_batch(
            entries: list[tuple[str, Any, Mapping[str, Any]]],
            directory: Path,
        ) -> Mapping[str, Any]:
            import numpy as np
            import tensorflow as tf

            bundle = {leaf_id: payload for leaf_id, payload, _ in entries}
            np.savez(str(directory / "tensors.npz"), **bundle)
            return {"tensorflow_version": tf.__version__}

        @staticmethod
        def read_batch(directory: Path, kind_meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import numpy as np
            import tensorflow as tf

            npz = np.load(directory / "tensors.npz", allow_pickle=False)

            def reader(leaf_id: str) -> Any:
                return tf.constant(np.array(npz[leaf_id]))

            return reader

    class TensorFlowSparseTensorSerializer(Serializer[Any]):
        """Directory serializer for ``tf.SparseTensor`` (3 sibling arrays)."""

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
