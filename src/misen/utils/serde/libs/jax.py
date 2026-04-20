"""Serializers for JAX arrays and dicts of JAX arrays."""

import importlib.util
from collections import OrderedDict
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

    class DictOfJaxArraysSerializer(Serializer[Any]):
        """Serialize ``dict[str, jax.Array]`` / ``OrderedDict[str, jax.Array]``.

        Targets flax param dicts and other flat jax.Array collections.  JAX
        arrays are converted to numpy and stored in a single ``.npz`` archive
        via ``np.savez`` — same backend as :class:`DictOfNdarraysSerializer`.
        The ``container`` field records whether the original was an
        ``OrderedDict`` so we can reconstruct it (``np.savez`` loads into a
        plain dict).

        Mixed or empty dicts and non-str keys fall through.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj) is not dict and type(obj) is not OrderedDict:
                return False
            if not obj:
                return False
            if not all(isinstance(k, str) for k in obj):
                return False
            import jax

            return all(isinstance(v, jax.Array) for v in obj.values())

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import jax
            import numpy as np

            np_dict = {k: np.asarray(v) for k, v in obj.items()}
            np.savez(str(directory / "data.npz"), **np_dict)
            return {
                "jax_version": jax.__version__,
                "container": "OrderedDict" if isinstance(obj, OrderedDict) else "dict",
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            import jax.numpy as jnp
            import numpy as np

            with np.load(directory / "data.npz", allow_pickle=False) as npz:
                np_dict = {k: npz[k] for k in npz.files}
            jax_dict = {k: jnp.asarray(v) for k, v in np_dict.items()}
            if meta.get("container") == "OrderedDict":
                return OrderedDict(jax_dict)
            return jax_dict

    jax_serializers = [JaxArraySerializer, DictOfJaxArraysSerializer]
    jax_serializers_by_type = {
        "jax.Array": JaxArraySerializer,
        # NOTE: DictOfJaxArraysSerializer is intentionally NOT listed here —
        # ``dict`` / ``OrderedDict`` are volatile_types on the serde registry.
    }
