"""Serializers for zarr Arrays and Groups.

zarr objects live in a hierarchical chunked store.  We round-trip
both kinds by writing a fresh on-disk store at the serde directory:
the array data, shape/dtype/chunks, and user attrs are preserved.
The compressor codec on the destination is whatever ``zarr.open``
chooses by default — different from the source if the source used a
non-default codec, but only the *encoding* of the bytes on disk
changes; reads return identical data and the user-facing API
(``arr[...]``, ``arr.shape``, ``arr.attrs``, etc.) behaves the same.

Compatible with zarr v2 and v3. Zarr v2 provides ``copy_all``; v3 does
not, so groups are copied recursively one array chunk at a time there.
"""

import importlib.util
import itertools
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.utils.serde.base import Serializer, translate_errors

__all__ = ["zarr_serializers", "zarr_serializers_by_type"]

zarr_serializers: list[type[Serializer]] = []
zarr_serializers_by_type: dict[str, type[Serializer]] = {}
_ZARR_V3 = 3


if importlib.util.find_spec("zarr") is not None:

    def _copy_attrs(source: Any, destination: Any) -> None:
        """Copy user attributes between Zarr nodes."""
        destination.attrs.update(source.attrs)

    def _copy_array_chunks(source: Any, destination: Any) -> None:
        """Copy an array without materializing its complete contents."""
        shape = tuple(source.shape)
        chunks = tuple(source.chunks)
        starts = (range(0, size, chunk) for size, chunk in zip(shape, chunks, strict=True))
        for offsets in itertools.product(*starts):
            region = tuple(
                slice(offset, min(offset + chunk, size))
                for offset, size, chunk in zip(offsets, shape, chunks, strict=True)
            )
            destination[region] = source[region]

    def _copy_group_v3(source: Any, destination: Any) -> None:
        """Recursively copy a group using the Zarr v3 synchronous API."""
        _copy_attrs(source, destination)
        for name, source_group in source.groups():
            _copy_group_v3(source_group, destination.create_group(name))
        for name, source_array in source.arrays():
            options: dict[str, Any] = {
                "shape": source_array.shape,
                "dtype": source_array.dtype,
                "chunks": source_array.chunks,
                "fill_value": source_array.fill_value,
            }
            dimension_names = getattr(source_array, "dimension_names", None)
            if dimension_names is not None:
                options["dimension_names"] = dimension_names
            destination_array = destination.create_array(name, **options)
            _copy_array_chunks(source_array, destination_array)
            _copy_attrs(source_array, destination_array)

    class ZarrArraySerializer(Serializer[Any]):
        """Serialize ``zarr.Array`` by writing a fresh zarr store.

        Materializes the array contents into a new on-disk store —
        chunks, dtype, and attrs are preserved.  The compressor on the
        destination is ``zarr.open``'s default; reads return the same
        data, but ``loaded.compressor`` may differ from
        ``original.compressor``.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            import zarr

            return isinstance(obj, zarr.Array)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import zarr

            store_path = str(directory / "data.zarr")
            with translate_errors(
                f"Could not encode Zarr array in {directory}", (KeyError, OSError, TypeError, ValueError, RuntimeError)
            ):
                dest = zarr.open(
                    store_path,
                    mode="w",
                    shape=obj.shape,
                    dtype=obj.dtype,
                    chunks=getattr(obj, "chunks", None),
                )
                dest[...] = obj[...]
                _copy_attrs(obj, dest)
            return {"zarr_version": zarr.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import zarr

            with translate_errors(
                f"Could not decode Zarr array in {directory}", (KeyError, OSError, TypeError, ValueError, RuntimeError)
            ):
                return zarr.open(str(directory / "data.zarr"), mode="r")

    class ZarrGroupSerializer(Serializer[Any]):
        """Serialize ``zarr.Group`` into a fresh on-disk store.

        Preserves nested groups, arrays, attrs, and chunks.  Like
        :class:`ZarrArraySerializer`, on-disk compressor codecs may
        differ from the source store; user-facing reads do not.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            import zarr

            return isinstance(obj, zarr.Group)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import zarr

            store_path = str(directory / "data.zarr")
            with translate_errors(
                f"Could not encode Zarr group in {directory}", (KeyError, OSError, TypeError, ValueError, RuntimeError)
            ):
                major_version = int(zarr.__version__.partition(".")[0])
                if major_version < _ZARR_V3:
                    dest = zarr.open_group(store_path, mode="w")
                    zarr.copy_all(obj, dest)
                    _copy_attrs(obj, dest)
                else:
                    source_format = getattr(getattr(obj, "metadata", None), "zarr_format", None)
                    dest = zarr.open_group(store_path, mode="w", zarr_format=source_format)
                    _copy_group_v3(obj, dest)
            return {"zarr_version": zarr.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import zarr

            with translate_errors(
                f"Could not decode Zarr group in {directory}", (KeyError, OSError, TypeError, ValueError, RuntimeError)
            ):
                return zarr.open_group(str(directory / "data.zarr"), mode="r")

    # The by-type fast path needs the concrete module paths, which differ
    # between zarr v2 (``zarr.core.Array`` / ``zarr.hierarchy.Group``) and
    # zarr v3 (``zarr.core.array.Array`` / ``zarr.core.group.Group``).
    # Compute them from the live classes rather than hard-coding.
    import zarr as _zarr

    from misen.utils.type_registry import qualified_type_name as _qname

    zarr_serializers = [ZarrArraySerializer, ZarrGroupSerializer]
    zarr_serializers_by_type = {
        _qname(_zarr.Array): ZarrArraySerializer,
        _qname(_zarr.Group): ZarrGroupSerializer,
    }
