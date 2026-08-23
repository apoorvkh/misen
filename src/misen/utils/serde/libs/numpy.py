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
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.exceptions import SerializationError
from misen.utils.serde.base import BaseSerializer, LeafSerializer, Serializer, translate_errors

__all__ = ["numpy_serializers", "numpy_serializers_by_type"]

numpy_serializers: list[type[BaseSerializer]] = []
numpy_serializers_by_type: dict[str, type[BaseSerializer]] = {}

if importlib.util.find_spec("numpy") is not None:
    import msgspec.msgpack
    import numpy as np

    from misen.utils.type_registry import qualified_type_name

    def _restore_dtype_descriptor(value: Any) -> Any:
        """Restore tuples erased when a structured dtype descriptor passes through msgpack."""
        if not isinstance(value, list):
            return value
        fields: list[tuple[Any, ...]] = []
        for raw_field in value:
            if not isinstance(raw_field, (list, tuple)) or len(raw_field) not in (2, 3):
                msg = f"Invalid NumPy dtype field descriptor: {raw_field!r}"
                raise TypeError(msg)
            name, dtype, *shape = raw_field
            if isinstance(name, list):
                name = tuple(name)
            if shape and isinstance(shape[0], list):
                shape[0] = tuple(shape[0])
            fields.append((name, _restore_dtype_descriptor(dtype), *shape))
        return fields

    def _scalar_dtype(entry: Mapping[str, Any]) -> np.dtype[Any]:
        """Decode the dtype recorded by either the current or legacy scalar format."""
        descriptor = entry["dtype_descr"] if "dtype_descr" in entry else entry["dtype"]
        return np.dtype(_restore_dtype_descriptor(descriptor))

    def _decode_scalar_entry(entry: Any, leaf_id: str) -> Any:
        """Decode and validate one current or legacy scalar payload."""
        if not isinstance(entry, Mapping):
            msg = f"NumPy scalar {leaf_id!r} has a non-mapping payload."
            raise TypeError(msg)
        dtype = _scalar_dtype(entry)
        if "data" not in entry:
            # v1 scalar payloads stored ``.item()`` under ``value``.
            return dtype.type(entry["value"])
        data = entry["data"]
        if not isinstance(data, bytes):
            msg = f"NumPy scalar {leaf_id!r} has a non-bytes data payload."
            raise TypeError(msg)
        if dtype.hasobject:
            msg = f"NumPy scalar {leaf_id!r} has an unsafe object-containing dtype {dtype!r}."
            raise ValueError(msg)
        if len(data) != dtype.itemsize:
            msg = f"NumPy scalar {leaf_id!r} has an incompatible binary payload for dtype {dtype!r}."
            raise ValueError(msg)
        return np.ndarray(shape=(), dtype=dtype, buffer=data)[()]

    class NumpyArraySerializer(LeafSerializer[Any]):
        """Batched leaf for ``numpy.ndarray`` — one ``arrays.npz`` per save."""

        leaf_kind = "ndarray"

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "numpy":
                return False
            # Strict ``type(obj) is np.ndarray`` — MaskedArray (a subclass)
            # has its own serializer because ``np.savez`` silently drops
            # the mask.
            return type(obj) is np.ndarray

        @classmethod
        def to_payload(cls, obj: Any) -> Any:
            """Reject arrays that would require pickle before writing the bundle."""
            if obj.dtype.hasobject:
                msg = f"Cannot serialize NumPy array with object-containing dtype {obj.dtype!r}."
                raise SerializationError(msg)
            return obj

        @staticmethod
        def write_batch(
            entries: list[tuple[str, Any, Mapping[str, Any]]],
            directory: Path,
        ) -> Mapping[str, Any]:
            bundle = {leaf_id: payload for leaf_id, payload, _ in entries}
            with translate_errors(
                f"Could not encode NumPy array bundle in {directory}", (OSError, TypeError, ValueError)
            ):
                np.savez(str(directory / "arrays.npz"), **bundle)
            return {"numpy_version": np.__version__}

        @staticmethod
        def read_batch(directory: Path, kind_meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            with translate_errors(
                f"Could not decode NumPy array bundle in {directory}",
                (OSError, EOFError, ValueError, zipfile.BadZipFile),
            ):
                npz = np.load(directory / "arrays.npz", allow_pickle=False)

            def reader(leaf_id: str) -> Any:
                # ``npz[leaf_id]`` returns a view backed by the mmap'd
                # archive; materialize with ``np.array`` so the reader
                # caller can close the archive without breaking readers.
                with translate_errors(
                    f"Could not decode NumPy array {leaf_id!r} in {directory}",
                    (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile),
                ):
                    return np.array(npz[leaf_id])

            return reader

    class NumpyScalarSerializer(LeafSerializer[Any]):
        """Batched leaf for numpy scalar values (e.g. ``np.float32(1.5)``).

        Stores dtype + raw scalar bytes for each scalar in a shared msgpack
        blob. This preserves values such as complex numbers, datetime units,
        extended-precision floats, and structured scalars without requiring
        msgpack to understand their Python ``.item()`` representation.
        """

        leaf_kind = "numpy_scalar"

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "numpy":
                return False
            return isinstance(obj, np.generic)

        @classmethod
        def to_payload(cls, obj: Any) -> Any:
            array = np.asarray(obj)
            if array.dtype.hasobject:
                msg = f"Cannot serialize NumPy scalar with object-containing dtype {array.dtype!r}."
                raise SerializationError(msg)
            with translate_errors(f"Could not encode NumPy scalar with dtype {array.dtype!r}", (TypeError, ValueError)):
                data = array.tobytes()
            payload: dict[str, Any] = {"dtype": array.dtype.str, "data": data}
            if array.dtype.fields is not None:
                payload["dtype_descr"] = array.dtype.descr
            return payload

        @staticmethod
        def write_batch(
            entries: list[tuple[str, Any, Mapping[str, Any]]],
            directory: Path,
        ) -> Mapping[str, Any]:
            blob = {leaf_id: payload for leaf_id, payload, _ in entries}
            with translate_errors(
                f"Could not encode NumPy scalar bundle in {directory}",
                (OSError, TypeError, ValueError, msgspec.EncodeError),
            ):
                encoded = msgspec.msgpack.encode(blob)
                (directory / "scalars.msgpack").write_bytes(encoded)
            return {"numpy_version": np.__version__}

        @staticmethod
        def read_batch(directory: Path, kind_meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            with translate_errors(
                f"Could not decode NumPy scalar bundle in {directory}", (OSError, msgspec.DecodeError)
            ):
                blob = msgspec.msgpack.decode((directory / "scalars.msgpack").read_bytes())

            def reader(leaf_id: str) -> Any:
                with translate_errors(
                    f"Could not decode NumPy scalar {leaf_id!r} in {directory}", (KeyError, TypeError, ValueError)
                ):
                    return _decode_scalar_entry(blob[leaf_id], leaf_id)

            return reader

    class NumpyMaskedArraySerializer(Serializer[Any]):
        """Directory serializer for ``numpy.ma.MaskedArray``.

        Writes ``data.npy``, ``mask.npy``, and a ``fill_value`` in the
        subdir meta.  MaskedArray subclasses ndarray but ``np.savez``
        drops the mask, so the leaf-batching path isn't safe here.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            return isinstance(obj, np.ma.MaskedArray)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            with translate_errors(
                f"Could not encode NumPy masked array in {directory}", (OSError, TypeError, ValueError)
            ):
                np.save(directory / "data.npy", obj.data, allow_pickle=False)
                np.save(directory / "mask.npy", obj.mask, allow_pickle=False)
                fill_value = obj.fill_value.item() if hasattr(obj.fill_value, "item") else obj.fill_value
            return {
                "numpy_version": np.__version__,
                "fill_value": fill_value,
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            with translate_errors(
                f"Could not decode NumPy masked array in {directory}", (OSError, EOFError, TypeError, ValueError)
            ):
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
    numpy_serializers_by_type = {
        qualified_type_name(np.ma.MaskedArray): NumpyMaskedArraySerializer,
        qualified_type_name(np.ndarray): NumpyArraySerializer,
        # These put their builtin ``str``/``bytes`` base before ``np.generic``
        # in the MRO, so register the concrete types to avoid builtin dispatch.
        qualified_type_name(np.str_): NumpyScalarSerializer,
        qualified_type_name(np.bytes_): NumpyScalarSerializer,
        qualified_type_name(np.generic): NumpyScalarSerializer,
    }
