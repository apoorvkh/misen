"""Handlers for scipy sparse matrices/arrays."""

import importlib.util
from typing import Any

from misen_hash.handler_base import HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import hash_msgspec, incremental_hash

__all__ = ["scipy_handlers", "scipy_handlers_by_type"]


scipy_handlers: HandlerTypeList = []
scipy_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("scipy") is not None:

    def _is_scipy_sparse(obj: Any) -> bool:
        obj_cls = type(obj)
        if obj_cls.__module__.split(".")[0] != "scipy":
            return False
        return any(
            base.__module__.startswith("scipy.sparse") and base.__name__ in {"spmatrix", "sparray"}
            for base in obj_cls.__mro__
        )

    class SciPySparseHandler(PrimitiveHandler):
        """Hash scipy sparse payload using COO triplets and metadata."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _is_scipy_sparse(obj)

        @staticmethod
        def digest(obj: Any) -> int:
            coo = obj.tocoo(copy=False)

            data = coo.data
            row = coo.row
            col = coo.col
            sparse_format = getattr(obj, "format", None)
            if sparse_format is None and hasattr(obj, "getformat"):
                sparse_format = obj.getformat()

            data_hash = incremental_hash(lambda sink: sink.write(memoryview(data).cast("B")))
            row_hash = incremental_hash(lambda sink: sink.write(memoryview(row).cast("B")))
            col_hash = incremental_hash(lambda sink: sink.write(memoryview(col).cast("B")))

            return hash_msgspec(
                (
                    sparse_format,
                    tuple(int(dim) for dim in obj.shape),
                    str(obj.dtype),
                    data_hash,
                    row_hash,
                    col_hash,
                )
            )

    scipy_handlers = [SciPySparseHandler]
    scipy_handlers_by_type = {
        "scipy.sparse._matrix.spmatrix": SciPySparseHandler,
        "scipy.sparse._base.sparray": SciPySparseHandler,
    }
