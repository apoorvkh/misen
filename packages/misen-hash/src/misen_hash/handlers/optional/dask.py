"""Handlers for dask collections."""

import importlib.util
from typing import Any

from misen_hash.handler_base import HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import hash_msgspec

__all__ = ["dask_handlers", "dask_handlers_by_type"]


dask_handlers: HandlerTypeList = []
dask_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("dask") is not None:

    class DaskCollectionHandler(PrimitiveHandler):
        """Hash dask collections by canonical dask tokenization."""

        @staticmethod
        def match(obj: Any) -> bool:
            import dask.base

            return dask.base.is_dask_collection(obj)

        @staticmethod
        def digest(obj: Any) -> int:
            import dask.base

            token = dask.base.tokenize(obj)
            return hash_msgspec(token)

    dask_handlers = [DaskCollectionHandler]
    dask_handlers_by_type = {}
