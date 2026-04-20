"""Serializers for numpy arrays, scalars, masked arrays, and arrays-in-dicts."""

import importlib.util
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import (
    Serializer,
    SerializerTypeRegistry,
)

__all__ = ["numpy_serializers", "numpy_serializers_by_type"]

numpy_serializers: list[type[Serializer]] = []
numpy_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("numpy") is not None:
    from pathlib import Path

    import msgspec.msgpack

    class NumpyArraySerializer(Serializer[Any]):
        """Serialize ``numpy.ndarray`` via the stable ``.npy`` format."""

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "numpy":
                return False
            import numpy as np

            return isinstance(obj, np.ndarray)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import numpy as np

            np.save(directory / "data.npy", obj, allow_pickle=False)
            return {"numpy_version": np.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import numpy as np

            return np.load(directory / "data.npy", allow_pickle=False)

    class NumpyScalarSerializer(Serializer[Any]):
        """Serialize numpy scalar values (e.g. ``np.float32(1.5)``)."""

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "numpy":
                return False
            import numpy as np

            return isinstance(obj, np.generic)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import numpy as np

            data = {"dtype": obj.dtype.str, "value": obj.item()}
            (directory / "data.msgpack").write_bytes(msgspec.msgpack.encode(data))
            return {"numpy_version": np.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import numpy as np

            raw = msgspec.msgpack.decode((directory / "data.msgpack").read_bytes())
            dtype = np.dtype(raw["dtype"])
            return dtype.type(raw["value"])

    class NumpyMaskedArraySerializer(Serializer[Any]):
        """Serialize ``numpy.ma.MaskedArray`` via separate data and mask ``.npy`` files."""

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
            fill_value = meta.get("fill_value")
            return np.ma.MaskedArray(data, mask=mask, fill_value=fill_value)

    class DictOfNdarraysSerializer(Serializer[Any]):
        """Serialize ``dict[str, np.ndarray]`` / ``OrderedDict[str, np.ndarray]``.

        Writes all arrays into a single ``.npz`` archive via ``np.savez``.
        ``np.savez`` requires string keys (they become filenames inside the
        archive), which matches the constraint of this serializer.  No extra
        dependency — ``.npz`` is numpy-native.

        Mixed dicts, non-str keys, empty dicts, and dict subclasses fall
        through to the next serializer (typically :class:`MsgpackSerializer`).
        """

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj) is not dict and type(obj) is not OrderedDict:
                return False
            if not obj:
                return False
            if not all(isinstance(k, str) for k in obj):
                return False
            import numpy as np

            # Plain ndarrays only — MaskedArray is a subclass but ``np.savez``
            # silently drops the mask, so refuse to match and let the caller
            # hit ``UnserializableTypeError`` from the msgpack fallback.
            return all(type(v) is np.ndarray for v in obj.values())

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import numpy as np

            # ``np.savez`` appends ``.npz`` if missing; writing to
            # ``data.npz`` already ends in ``.npz`` so no double suffix.
            np.savez(str(directory / "data.npz"), **obj)
            return {
                "numpy_version": np.__version__,
                "container": "OrderedDict" if isinstance(obj, OrderedDict) else "dict",
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            import numpy as np

            with np.load(directory / "data.npz", allow_pickle=False) as npz:
                # ``npz.files`` preserves the order arrays were saved in,
                # which matches the insertion order of the original dict.
                loaded = {k: npz[k] for k in npz.files}
            if meta.get("container") == "OrderedDict":
                return OrderedDict(loaded)
            return loaded

    # ``NumpyMaskedArraySerializer`` must precede ``NumpyArraySerializer`` —
    # MaskedArray is a subclass of ndarray, so the linear-scan fallback
    # picks whichever matches first.  ``DictOfNdarraysSerializer`` order
    # doesn't matter (its match is disjoint from the other three), and it
    # must precede ``MsgpackSerializer`` — enforced by ``numpy_serializers``
    # appearing before ``stdlib_serializers`` in ``libs/__init__.py``.
    numpy_serializers = [
        NumpyMaskedArraySerializer,
        NumpyArraySerializer,
        NumpyScalarSerializer,
        DictOfNdarraysSerializer,
    ]
    numpy_serializers_by_type = {
        "numpy.ma.core.MaskedArray": NumpyMaskedArraySerializer,
        "numpy.ndarray": NumpyArraySerializer,
        "numpy.generic": NumpyScalarSerializer,
        # NOTE: DictOfNdarraysSerializer is intentionally NOT listed here —
        # dicts/OrderedDicts are ``volatile_types`` on the serde registry
        # and dispatch through the linear-scan ``match`` path.
    }
