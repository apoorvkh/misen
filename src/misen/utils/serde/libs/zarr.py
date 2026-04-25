"""Serializers for zarr Arrays and Groups.

zarr objects live in a hierarchical chunked store.  We round-trip
both kinds by writing a fresh on-disk store at the serde directory:
the array data, shape/dtype/chunks, and user attrs are preserved.
The compressor codec on the destination is whatever ``zarr.open``
chooses by default — different from the source if the source used a
non-default codec, but only the *encoding* of the bytes on disk
changes; reads return identical data and the user-facing API
(``arr[...]``, ``arr.shape``, ``arr.attrs``, etc.) behaves the same.

Compatible with zarr v2 and v3 — both export ``zarr.Array`` /
``zarr.Group`` and accept ``zarr.open`` / ``zarr.open_group`` /
``zarr.copy_all`` with stable semantics.
"""

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.utils.serde.base import Serializer

__all__ = ["zarr_serializers", "zarr_serializers_by_type"]

zarr_serializers: list[type[Serializer]] = []
zarr_serializers_by_type: dict[str, type[Serializer]] = {}


if importlib.util.find_spec("zarr") is not None:

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
            dest = zarr.open(
                store_path,
                mode="w",
                shape=obj.shape,
                dtype=obj.dtype,
                chunks=getattr(obj, "chunks", None),
            )
            dest[...] = obj[...]
            for key, val in obj.attrs.items():
                dest.attrs[key] = val
            return {"zarr_version": zarr.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import zarr

            return zarr.open(str(directory / "data.zarr"), mode="r")

    class ZarrGroupSerializer(Serializer[Any]):
        """Serialize ``zarr.Group`` via :func:`zarr.copy_all` into a fresh store.

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
            dest = zarr.open_group(store_path, mode="w")
            zarr.copy_all(obj, dest)
            for key, val in obj.attrs.items():
                dest.attrs[key] = val
            return {"zarr_version": zarr.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import zarr

            return zarr.open_group(str(directory / "data.zarr"), mode="r")

    # The by-type fast path needs the concrete module paths, which differ
    # between zarr v2 (``zarr.core.Array`` / ``zarr.hierarchy.Group``) and
    # zarr v3 (``zarr.core.array.Array`` / ``zarr.core.group.Group``).
    # Compute them from the live classes rather than hard-coding.
    from misen.utils.type_registry import qualified_type_name as _qname

    import zarr as _zarr

    zarr_serializers = [ZarrArraySerializer, ZarrGroupSerializer]
    zarr_serializers_by_type = {
        _qname(_zarr.Array): ZarrArraySerializer,
        _qname(_zarr.Group): ZarrGroupSerializer,
    }
