"""Numpy v2 serializers.

- :class:`NumpyArraySerializer` and :class:`NumpyScalarSerializer` are
  :class:`LeafSerializer` subclasses: all arrays (resp. scalars) in a
  single save get batched into one ``arrays.npz`` (resp. one shared
  msgpack blob).  A dict-of-ndarrays — or a dict-of-dict-of-ndarrays —
  thus packs into one npz regardless of nesting depth, subsuming v1's
  special-case ``DictOfNdarraysSerializer``.

- :class:`NumpyMaskedArraySerializer` is a :class:`Serializer`
  because masked arrays carry a sibling mask array; the
  leaf-batching model doesn't fit.
"""

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.utils.serde.base import LeafSerializer, Serializer

__all__ = ["numpy_serializers", "numpy_serializers_by_type"]

numpy_serializers: list[type[Serializer]] = []
numpy_serializers_by_type: dict[str, type[Serializer]] = {}

if importlib.util.find_spec("numpy") is not None:
    import msgspec.msgpack

    class NumpyArraySerializer(LeafSerializer[Any]):
        """Batched leaf for ``numpy.ndarray`` — one ``arrays.npz`` per save."""

        leaf_kind = "ndarray"

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "numpy":
                return False
            import numpy as np

            # Strict ``type(obj) is np.ndarray`` — MaskedArray (a subclass)
            # has its own serializer because ``np.savez`` silently drops
            # the mask.
            return type(obj) is np.ndarray

        @staticmethod
        def write_batch(
            entries: list[tuple[str, Any, Mapping[str, Any]]],
            directory: Path,
        ) -> Mapping[str, Any]:
            import numpy as np

            bundle = {leaf_id: payload for leaf_id, payload, _ in entries}
            np.savez(str(directory / "arrays.npz"), **bundle)
            return {"numpy_version": np.__version__}

        @staticmethod
        def read_batch(directory: Path, kind_meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import numpy as np

            npz = np.load(directory / "arrays.npz", allow_pickle=False)

            def reader(leaf_id: str) -> Any:
                # ``npz[leaf_id]`` returns a view backed by the mmap'd
                # archive; materialize with ``np.array`` so the reader
                # caller can close the archive without breaking readers.
                return np.array(npz[leaf_id])

            return reader

    class NumpyScalarSerializer(LeafSerializer[Any]):
        """Batched leaf for numpy scalar values (e.g. ``np.float32(1.5)``).

        Stores dtype + python value for each scalar in a shared msgpack
        blob; dtype is reconstructed on read so the original numpy type
        round-trips.
        """

        leaf_kind = "numpy_scalar"

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "numpy":
                return False
            import numpy as np

            return isinstance(obj, np.generic)

        @classmethod
        def to_payload(cls, obj: Any) -> Any:
            return {"dtype": obj.dtype.str, "value": obj.item()}

        @staticmethod
        def write_batch(
            entries: list[tuple[str, Any, Mapping[str, Any]]],
            directory: Path,
        ) -> Mapping[str, Any]:
            import numpy as np

            blob = {leaf_id: payload for leaf_id, payload, _ in entries}
            (directory / "scalars.msgpack").write_bytes(msgspec.msgpack.encode(blob))
            return {"numpy_version": np.__version__}

        @staticmethod
        def read_batch(directory: Path, kind_meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import numpy as np

            blob = msgspec.msgpack.decode((directory / "scalars.msgpack").read_bytes())

            def reader(leaf_id: str) -> Any:
                entry = blob[leaf_id]
                return np.dtype(entry["dtype"]).type(entry["value"])

            return reader

    class NumpyMaskedArraySerializer(Serializer[Any]):
        """Directory serializer for ``numpy.ma.MaskedArray``.

        Writes ``data.npy``, ``mask.npy``, and a ``fill_value`` in the
        subdir meta.  MaskedArray subclasses ndarray but ``np.savez``
        drops the mask, so the leaf-batching path isn't safe here.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            import numpy as np

            return isinstance(obj, np.ma.MaskedArray)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import numpy as np

            np.save(directory / "data.npy", obj.data, allow_pickle=False)
            np.save(directory / "mask.npy", obj.mask, allow_pickle=False)
            fill_value = obj.fill_value.item() if hasattr(obj.fill_value, "item") else obj.fill_value
            return {
                "numpy_version": np.__version__,
                "fill_value": fill_value,
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            import numpy as np

            data = np.load(directory / "data.npy", allow_pickle=False)
            mask = np.load(directory / "mask.npy", allow_pickle=False)
            return np.ma.MaskedArray(data, mask=mask, fill_value=meta.get("fill_value"))

    # ``NumpyMaskedArraySerializer`` must come before ``NumpyArraySerializer``
    # — MaskedArray is an ndarray subclass, so linear-scan dispatch picks
    # whichever matches first.  (Strict ``type(obj) is np.ndarray`` on the
    # array serializer's match makes this ordering defensive rather than
    # required, but the convention is preserved.)
    numpy_serializers = [
        NumpyMaskedArraySerializer,
        NumpyArraySerializer,
        NumpyScalarSerializer,
    ]

    # Build the by-type fast path from the *actual* qualified names of
    # the numpy classes.  Historically v1 hard-coded
    # ``"numpy.ma.core.MaskedArray"`` but modern numpy reports
    # ``"numpy.ma.MaskedArray"`` for ``__module__.__qualname__`` — a
    # mismatch there silently routes MaskedArray to
    # :class:`NumpyArraySerializer` via the MRO walk (``np.ndarray``
    # is the next base) and drops the mask.  Importing the classes
    # here and letting :func:`qualified_type_name` compute the names
    # keeps this robust across numpy versions.
    import numpy as _np
    from misen.utils.type_registry import qualified_type_name as _qname

    numpy_serializers_by_type = {
        _qname(_np.ma.MaskedArray): NumpyMaskedArraySerializer,
        _qname(_np.ndarray): NumpyArraySerializer,
        _qname(_np.generic): NumpyScalarSerializer,
    }
