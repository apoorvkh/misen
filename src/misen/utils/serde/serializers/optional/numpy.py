"""Serializers for numpy arrays, scalars, and masked arrays."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["numpy_serializers", "numpy_serializers_by_type"]

numpy_serializers: SerializerTypeList = []
numpy_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("numpy") is not None:
    from pathlib import Path

    import msgspec.msgpack

    class NumpyArraySerializer(Serializer[Any]):
        """Serialize ``numpy.ndarray`` via the stable ``.npy`` format."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "numpy":
                return False
            import numpy as np

            return isinstance(obj, np.ndarray)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import numpy as np

            np.save(directory / "data.npy", obj, allow_pickle=False)
            write_meta(directory, NumpyArraySerializer, numpy_version=np.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            import numpy as np

            return np.load(directory / "data.npy", allow_pickle=False)

    class NumpyScalarSerializer(Serializer[Any]):
        """Serialize numpy scalar values (e.g. ``np.float32(1.5)``)."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "numpy":
                return False
            import numpy as np

            return isinstance(obj, np.generic)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import numpy as np

            data = {"dtype": obj.dtype.str, "value": obj.item()}
            (directory / "data.msgpack").write_bytes(msgspec.msgpack.encode(data))
            write_meta(directory, NumpyScalarSerializer, numpy_version=np.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            import numpy as np

            raw = msgspec.msgpack.decode((directory / "data.msgpack").read_bytes())
            dtype = np.dtype(raw["dtype"])
            return dtype.type(raw["value"])

    class NumpyMaskedArraySerializer(Serializer[Any]):
        """Serialize ``numpy.ma.MaskedArray`` via separate data and mask ``.npy`` files."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import numpy as np

            return isinstance(obj, np.ma.MaskedArray)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import numpy as np

            np.save(directory / "data.npy", obj.data, allow_pickle=False)
            np.save(directory / "mask.npy", obj.mask, allow_pickle=False)
            fill_value = obj.fill_value.item() if hasattr(obj.fill_value, "item") else obj.fill_value
            write_meta(
                directory,
                NumpyMaskedArraySerializer,
                numpy_version=np.__version__,
                fill_value=fill_value,
            )

        @staticmethod
        def load(directory: Path) -> Any:
            import numpy as np

            from misen.utils.serde.serializer_base import read_meta

            data = np.load(directory / "data.npy", allow_pickle=False)
            mask = np.load(directory / "mask.npy", allow_pickle=False)
            meta = read_meta(directory)
            fill_value = meta.get("fill_value") if meta else None
            return np.ma.MaskedArray(data, mask=mask, fill_value=fill_value)

    # MaskedArray must come before ndarray (it's a subclass).
    numpy_serializers = [NumpyMaskedArraySerializer, NumpyArraySerializer, NumpyScalarSerializer]
    numpy_serializers_by_type = {
        "numpy.ma.core.MaskedArray": NumpyMaskedArraySerializer,
        "numpy.ndarray": NumpyArraySerializer,
        "numpy.generic": NumpyScalarSerializer,
    }
