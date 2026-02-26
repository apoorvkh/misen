"""Handlers for tensorflow tensors, variables, and Keras models."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import hash_msgspec, incremental_hash

__all__ = ["tensorflow_handlers", "tensorflow_handlers_by_type"]


tensorflow_handlers: HandlerTypeList = []
tensorflow_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("tensorflow") is not None:
    _numpy_available = importlib.util.find_spec("numpy") is not None

    def _has_keras_model_base(obj: Any) -> bool:
        return any(
            base.__name__ == "Model" and base.__module__.split(".")[0] in {"keras", "tensorflow"}
            for base in type(obj).__mro__
        )

    class TensorFlowVariableHandler(PrimitiveHandler):
        """Hash tensorflow variables by value, dtype, shape, and metadata."""

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "tensorflow":
                return False

            import tensorflow as tf

            return isinstance(obj, tf.Variable)

        @staticmethod
        def digest(obj: Any) -> int:
            import tensorflow as tf

            tensor = tf.convert_to_tensor(obj)
            serialized = bytes(tf.io.serialize_tensor(tensor).numpy())
            payload_hash = incremental_hash(lambda sink: sink.write(serialized))
            return hash_msgspec(
                (
                    tensor.dtype.name,
                    tuple(int(dim) for dim in tensor.shape),
                    payload_hash,
                    obj.name,
                    bool(obj.trainable),
                )
            )

    class TensorFlowTensorHandler(PrimitiveHandler):
        """Hash tensorflow tensors by dtype, shape, and serialized bytes."""

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "tensorflow":
                return False

            import tensorflow as tf

            return tf.is_tensor(obj)

        @staticmethod
        def digest(obj: Any) -> int:
            import tensorflow as tf

            tensor = tf.convert_to_tensor(obj)
            serialized = bytes(tf.io.serialize_tensor(tensor).numpy())
            payload_hash = incremental_hash(lambda sink: sink.write(serialized))
            return hash_msgspec((tensor.dtype.name, tuple(int(dim) for dim in tensor.shape), payload_hash))

    class KerasModelHandler(CollectionHandler):
        """Hash Keras model class/config plus weight payload."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_keras_model_base(obj)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            if not _numpy_available:
                msg = "numpy must be installed if using tensorflow objects in misen. Please `pip install numpy`."
                raise ImportError(msg)

            config = None
            if hasattr(obj, "to_json"):
                try:
                    config = obj.to_json()
                except ValueError:
                    config = None

            weight_entries: list[tuple[str, str, tuple[int, ...], int]] = []
            for variable in getattr(obj, "weights", []):
                array = variable.numpy()
                payload = array.tobytes()
                payload_hash = incremental_hash(lambda sink, payload=payload: sink.write(payload))
                weight_entries.append(
                    (
                        variable.name,
                        str(variable.dtype),
                        tuple(int(dim) for dim in variable.shape),
                        payload_hash,
                    )
                )

            weight_entries.sort(key=lambda item: item[0])

            return [
                config,
                weight_entries,
            ]

    tensorflow_handlers = [
        TensorFlowVariableHandler,
        TensorFlowTensorHandler,
        KerasModelHandler,
    ]
    tensorflow_handlers_by_type = {
        "tensorflow.python.ops.resource_variable_ops.ResourceVariable": TensorFlowVariableHandler,
        "tensorflow.python.framework.tensor.Tensor": TensorFlowTensorHandler,
        "keras.src.models.model.Model": KerasModelHandler,
        "tensorflow.python.keras.engine.training.Model": KerasModelHandler,
    }
