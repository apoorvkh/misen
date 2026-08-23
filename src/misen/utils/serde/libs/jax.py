"""JAX array v2 serializer — batched ``.npz`` via numpy conversion."""

import importlib.util
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.utils.serde.base import BaseSerializer, LeafSerializer, translate_errors

__all__ = ["jax_serializers", "jax_serializers_by_type"]

jax_serializers: list[type[BaseSerializer]] = []
jax_serializers_by_type: dict[str, type[BaseSerializer]] = {}

if importlib.util.find_spec("jax") is not None and importlib.util.find_spec("numpy") is not None:
    import jax as _jax

    from misen.utils.type_registry import qualified_type_name as _qname

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
            import jax

            return isinstance(obj, jax.Array)

        @classmethod
        def to_payload(cls, obj: Any) -> Any:
            import numpy as np

            with translate_errors(
                "Could not convert JAX array for serialization", (TypeError, ValueError, RuntimeError)
            ):
                return np.asarray(obj)

        @staticmethod
        def write_batch(
            entries: list[tuple[str, Any, Mapping[str, Any]]],
            directory: Path,
        ) -> Mapping[str, Any]:
            import jax
            import numpy as np

            bundle = {leaf_id: payload for leaf_id, payload, _ in entries}
            with translate_errors(
                f"Could not encode JAX array bundle in {directory}", (OSError, TypeError, ValueError)
            ):
                np.savez(str(directory / "arrays.npz"), **bundle)
            return {"jax_version": jax.__version__}

        @staticmethod
        def read_batch(directory: Path, kind_meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import jax.numpy as jnp
            import numpy as np

            with translate_errors(
                f"Could not decode JAX array bundle in {directory}",
                (OSError, EOFError, ValueError, zipfile.BadZipFile),
            ):
                npz = np.load(directory / "arrays.npz", allow_pickle=False)

            def reader(leaf_id: str) -> Any:
                with translate_errors(
                    f"Could not decode JAX array {leaf_id!r} in {directory}",
                    (OSError, KeyError, TypeError, ValueError, RuntimeError, zipfile.BadZipFile),
                ):
                    return jnp.asarray(np.array(npz[leaf_id]))

            return reader

    jax_serializers = [JaxArraySerializer]
    jax_serializers_by_type = {_qname(_jax.Array): JaxArraySerializer}
