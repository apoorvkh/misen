"""TensorFlow v2 serializers — batched tensors, per-instance sparse tensors."""

import importlib.util
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.utils.serde.base import BaseSerializer, LeafSerializer, Serializer, translate_errors

__all__ = ["tensorflow_serializers", "tensorflow_serializers_by_type"]

tensorflow_serializers: list[type[BaseSerializer]] = []
tensorflow_serializers_by_type: dict[str, type[BaseSerializer]] = {}

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
            import tensorflow as tf

            with translate_errors(
                "Could not convert TensorFlow tensor for serialization",
                (AttributeError, TypeError, ValueError, RuntimeError, tf.errors.OpError),
            ):
                return obj.numpy()

        @staticmethod
        def write_batch(
            entries: list[tuple[str, Any, Mapping[str, Any]]],
            directory: Path,
        ) -> Mapping[str, Any]:
            import numpy as np
            import tensorflow as tf

            bundle = {leaf_id: payload for leaf_id, payload, _ in entries}
            with translate_errors(
                f"Could not encode TensorFlow tensor bundle in {directory}", (OSError, TypeError, ValueError)
            ):
                np.savez(str(directory / "tensors.npz"), **bundle)
            return {"tensorflow_version": tf.__version__}

        @staticmethod
        def read_batch(directory: Path, kind_meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import numpy as np
            import tensorflow as tf

            with translate_errors(
                f"Could not decode TensorFlow tensor bundle in {directory}",
                (OSError, EOFError, ValueError, zipfile.BadZipFile),
            ):
                npz = np.load(directory / "tensors.npz", allow_pickle=False)

            def reader(leaf_id: str) -> Any:
                with translate_errors(
                    f"Could not decode TensorFlow tensor {leaf_id!r} in {directory}",
                    (
                        OSError,
                        KeyError,
                        TypeError,
                        ValueError,
                        RuntimeError,
                        zipfile.BadZipFile,
                        tf.errors.OpError,
                    ),
                ):
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

            with translate_errors(
                f"Could not encode TensorFlow sparse tensor in {directory}",
                (OSError, AttributeError, TypeError, ValueError, RuntimeError, tf.errors.OpError),
            ):
                np.save(directory / "indices.npy", obj.indices.numpy(), allow_pickle=False)
                np.save(directory / "values.npy", obj.values.numpy(), allow_pickle=False)
                dense_shape = list(obj.dense_shape.numpy())
            return {
                "tensorflow_version": tf.__version__,
                "dense_shape": dense_shape,
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            import numpy as np
            import tensorflow as tf

            with translate_errors(
                f"Could not decode TensorFlow sparse tensor in {directory}",
                (OSError, EOFError, KeyError, TypeError, ValueError, RuntimeError, tf.errors.OpError),
            ):
                indices = np.load(directory / "indices.npy", allow_pickle=False)
                values = np.load(directory / "values.npy", allow_pickle=False)
                return tf.SparseTensor(indices=indices, values=values, dense_shape=meta["dense_shape"])

    tensorflow_serializers = [TensorFlowSparseTensorSerializer, TensorFlowTensorSerializer]
    tensorflow_serializers_by_type = {
        "tensorflow.python.framework.ops.EagerTensor": TensorFlowTensorSerializer,
        "tensorflow.python.framework.sparse_tensor.SparseTensor": TensorFlowSparseTensorSerializer,
    }
