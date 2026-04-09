"""Serializer for JAX arrays via numpy .npy format."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["jax_serializers", "jax_serializers_by_type"]

jax_serializers: SerializerTypeList = []
jax_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("jax") is not None and importlib.util.find_spec("numpy") is not None:
    from pathlib import Path

    class JaxArraySerializer(Serializer[Any]):
        """Serialize JAX arrays by converting to numpy and saving as ``.npy``."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "jax":
                return False
            import jax

            return isinstance(obj, jax.Array)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import jax
            import numpy as np

            arr = np.asarray(obj)
            np.save(directory / "data.npy", arr, allow_pickle=False)
            write_meta(
                directory,
                JaxArraySerializer,
                jax_version=jax.__version__,
                dtype=str(arr.dtype),
                shape=list(arr.shape),
            )

        @staticmethod
        def load(directory: Path) -> Any:
            import jax.numpy as jnp
            import numpy as np

            arr = np.load(directory / "data.npy", allow_pickle=False)
            return jnp.asarray(arr)

    jax_serializers = [JaxArraySerializer]
    jax_serializers_by_type = {"jax.Array": JaxArraySerializer}
