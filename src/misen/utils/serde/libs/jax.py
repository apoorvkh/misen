"""Serializer for JAX arrays via numpy .npy format."""

import importlib.util
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import (
    Serializer,
    SerializerTypeRegistry,
)

__all__ = ["jax_serializers", "jax_serializers_by_type"]

jax_serializers: list[type[Serializer]] = []
jax_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("jax") is not None and importlib.util.find_spec("numpy") is not None:
    from pathlib import Path

    class JaxArraySerializer(Serializer[Any]):
        """Serialize JAX arrays by converting to numpy and saving as ``.npy``."""

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "jax":
                return False
            import jax

            return isinstance(obj, jax.Array)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import jax
            import numpy as np

            arr = np.asarray(obj)
            np.save(directory / "data.npy", arr, allow_pickle=False)
            return {
                "jax_version": jax.__version__,
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import jax.numpy as jnp
            import numpy as np

            arr = np.load(directory / "data.npy", allow_pickle=False)
            return jnp.asarray(arr)

    jax_serializers = [JaxArraySerializer]
    jax_serializers_by_type = {"jax.Array": JaxArraySerializer}
