"""Handlers for jax arrays."""

import importlib.util
from typing import Any

from misen_hash.handler_base import HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import hash_msgspec, incremental_hash

__all__ = ["jax_handlers", "jax_handlers_by_type"]


jax_handlers: HandlerTypeList = []
jax_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("jax") is not None:
    _numpy_available = importlib.util.find_spec("numpy") is not None

    def _is_jax_array(obj: Any) -> bool:
        if type(obj).__module__.split(".")[0] not in {"jax", "jaxlib"}:
            return False

        import jax

        array_type = getattr(jax, "Array", None)
        if array_type is not None and isinstance(obj, array_type):
            return True
        return hasattr(obj, "__jax_array__") and hasattr(obj, "shape") and hasattr(obj, "dtype")

    class JaxArrayHandler(PrimitiveHandler):
        """Hash jax arrays by dtype/shape/content."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _is_jax_array(obj)

        @staticmethod
        def digest(obj: Any) -> int:
            if not _numpy_available:
                msg = "numpy must be installed if using jax objects in misen. Please `pip install numpy`."
                raise ImportError(msg)

            import numpy as np

            array = np.asarray(obj)
            shape = tuple(int(dim) for dim in array.shape)

            if array.dtype.hasobject:
                return hash_msgspec((str(array.dtype), shape, array.reshape(-1).tolist()))

            contiguous = np.ascontiguousarray(array)
            payload_hash = incremental_hash(lambda sink: sink.write(contiguous.tobytes()))
            return hash_msgspec((str(contiguous.dtype), shape, payload_hash))

    jax_handlers = [JaxArrayHandler]
    jax_handlers_by_type = {"jax.Array": JaxArrayHandler}
