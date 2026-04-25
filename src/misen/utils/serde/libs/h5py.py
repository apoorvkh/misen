"""Serializer for h5py File objects.

:class:`h5py.File` is a filesystem-backed container.  We copy the
underlying HDF5 bytes verbatim rather than walking the tree, which
preserves datasets, groups, attributes, compression filters, and
dataset layouts without interpretation.

The reader returns an ``h5py.File`` opened in read-only mode.  Callers
own the lifetime of the handle (``f.close()`` or ``with`` block).
"""

import importlib.util
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.utils.serde.base import Serializer

__all__ = ["h5py_serializers", "h5py_serializers_by_type"]

h5py_serializers: list[type[Serializer]] = []
h5py_serializers_by_type: dict[str, type[Serializer]] = {}


if importlib.util.find_spec("h5py") is not None:

    class H5pyFileSerializer(Serializer[Any]):
        """Serialize ``h5py.File`` by copying the backing HDF5 file.

        Round-trips any File whose backing store is an on-disk path
        (the common case).  In-memory / driver-specific backends are
        flushed to a new on-disk file via ``h5py.File.copy``.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            import h5py

            return isinstance(obj, h5py.File)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import h5py

            dest = directory / "data.h5"
            filename = getattr(obj, "filename", None)
            # Fast path: the File is backed by a real path we can copy.
            if filename and Path(filename).exists():
                # Flush any pending writes before copying the bytes.
                obj.flush()
                shutil.copyfile(filename, dest)
            else:
                # Fallback: walk the object tree into a fresh file.
                with h5py.File(dest, "w") as out:
                    for name in obj:
                        obj.copy(name, out)
                    for key, val in obj.attrs.items():
                        out.attrs[key] = val
            return {"h5py_version": h5py.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import h5py

            return h5py.File(directory / "data.h5", "r")

    h5py_serializers = [H5pyFileSerializer]
    h5py_serializers_by_type = {"h5py._hl.files.File": H5pyFileSerializer}
