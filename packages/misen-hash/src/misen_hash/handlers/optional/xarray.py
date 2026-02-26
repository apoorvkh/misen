"""Handlers for xarray objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry

__all__ = ["xarray_handlers", "xarray_handlers_by_type"]


xarray_handlers: HandlerTypeList = []
xarray_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("xarray") is not None:

    def _has_xarray_base(obj: Any, type_name: str) -> bool:
        return any(
            base.__name__ == type_name and base.__module__.split(".")[0] == "xarray"
            for base in type(obj).__mro__
        )

    class XarrayDataArrayHandler(CollectionHandler):
        """Hash xarray DataArray objects by serialized dictionary form."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_xarray_base(obj, "DataArray")

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            return [obj.to_dict(data=True)]


    class XarrayDatasetHandler(CollectionHandler):
        """Hash xarray Dataset objects by serialized dictionary form."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_xarray_base(obj, "Dataset")

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            return [obj.to_dict(data=True)]

    xarray_handlers = [
        XarrayDataArrayHandler,
        XarrayDatasetHandler,
    ]
    xarray_handlers_by_type = {
        "xarray.core.dataarray.DataArray": XarrayDataArrayHandler,
        "xarray.core.dataset.Dataset": XarrayDatasetHandler,
    }
