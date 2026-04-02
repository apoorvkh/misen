"""Handlers for declarative numpy dtype and scalar values."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import hash_msgspec, incremental_hash

__all__ = ["numpy_handlers", "numpy_handlers_by_type"]

numpy_handlers: HandlerTypeList = []
numpy_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("numpy") is not None:

    class NumpyDTypeHandler(PrimitiveHandler):
        """Hash numpy dtype objects by structural descriptor."""

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "numpy":
                return False

            import numpy as np

            return isinstance(obj, np.dtype)

        @staticmethod
        def digest(obj: Any) -> int:
            import numpy as np

            dtype = np.dtype(obj)
            return hash_msgspec((dtype.str, dtype.descr))


    class NumpyScalarHandler(CollectionHandler):
        """Hash numpy scalar values with dtype fidelity."""

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "numpy":
                return False

            import numpy as np

            return isinstance(obj, np.generic)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            scalar = obj

            if scalar.dtype.hasobject:
                return [scalar.dtype.str, scalar.item()]

            payload_hash = incremental_hash(lambda sink: sink.write(scalar.tobytes()))
            return [scalar.dtype.str, payload_hash]


    numpy_handlers = [
        NumpyDTypeHandler,
        NumpyScalarHandler,
    ]
    numpy_handlers_by_type = {
        "numpy.dtype": NumpyDTypeHandler,
        "numpy.generic": NumpyScalarHandler,
    }
