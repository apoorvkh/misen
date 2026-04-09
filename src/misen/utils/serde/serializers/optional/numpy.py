"""Serializers for numpy arrays and scalars."""

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

    numpy_serializers = [NumpyArraySerializer, NumpyScalarSerializer]
    numpy_serializers_by_type = {
        "numpy.ndarray": NumpyArraySerializer,
        "numpy.generic": NumpyScalarSerializer,
    }
