"""JAX array v2 serializer — batched ``.npz`` via numpy conversion."""

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.utils.serde.base import LeafSerializer, Serializer

__all__ = ["jax_serializers", "jax_serializers_by_type"]

jax_serializers: list[type[Serializer]] = []
jax_serializers_by_type: dict[str, type[Serializer]] = {}

if importlib.util.find_spec("jax") is not None and importlib.util.find_spec("numpy") is not None:

    class JaxArraySerializer(LeafSerializer[Any]):
        """Batched leaf for ``jax.Array`` — packed into one ``arrays.npz``.

        Arrays are converted to numpy on write and back on read.  Like
        the numpy serializer, a deeply nested dict of jax arrays
        collapses into a single npz without needing the
        ``DictOfJaxArraysSerializer`` special case from v1.
        """

        leaf_kind = "jax_array"

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "jax":
                return False
            import jax

            return isinstance(obj, jax.Array)

        @classmethod
        def to_payload(cls, obj: Any) -> Any:
            import numpy as np

            return np.asarray(obj)

        @staticmethod
        def write_batch(
            entries: list[tuple[str, Any, Mapping[str, Any]]],
            directory: Path,
        ) -> Mapping[str, Any]:
            import jax
            import numpy as np

            bundle = {leaf_id: payload for leaf_id, payload, _ in entries}
            np.savez(str(directory / "arrays.npz"), **bundle)
            return {"jax_version": jax.__version__}

        @staticmethod
        def read_batch(directory: Path, kind_meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import jax.numpy as jnp
            import numpy as np

            npz = np.load(directory / "arrays.npz", allow_pickle=False)

            def reader(leaf_id: str) -> Any:
                return jnp.asarray(np.array(npz[leaf_id]))

            return reader

    jax_serializers = [JaxArraySerializer]
    jax_serializers_by_type = {"jax.Array": JaxArraySerializer}
