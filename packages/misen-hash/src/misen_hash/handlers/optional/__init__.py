"""Aggregate optional handlers from per-library modules."""

from misen_hash.handler_base import HandlerTypeList, HandlerTypeRegistry
from misen_hash.handlers.optional.attrs import attrs_handlers, attrs_handlers_by_type
from misen_hash.handlers.optional.msgspec import msgspec_handlers, msgspec_handlers_by_type
from misen_hash.handlers.optional.networkx import networkx_handlers, networkx_handlers_by_type
from misen_hash.handlers.optional.numpy import numpy_handlers, numpy_handlers_by_type
from misen_hash.handlers.optional.pandas import pandas_handlers, pandas_handlers_by_type
from misen_hash.handlers.optional.pillow import pillow_handlers, pillow_handlers_by_type
from misen_hash.handlers.optional.polars import polars_handlers, polars_handlers_by_type
from misen_hash.handlers.optional.pyarrow import pyarrow_handlers, pyarrow_handlers_by_type
from misen_hash.handlers.optional.pydantic import pydantic_handlers, pydantic_handlers_by_type
from misen_hash.handlers.optional.scipy import scipy_handlers, scipy_handlers_by_type
from misen_hash.handlers.optional.sklearn import sklearn_handlers, sklearn_handlers_by_type
from misen_hash.handlers.optional.torch import torch_handlers, torch_handlers_by_type
from misen_hash.handlers.optional.xarray import xarray_handlers, xarray_handlers_by_type

__all__ = ["optional_handlers", "optional_handlers_by_type"]

optional_handlers: HandlerTypeList = [
    *msgspec_handlers,
    *attrs_handlers,
    *numpy_handlers,
    *pandas_handlers,
    *pydantic_handlers,
    *polars_handlers,
    *pyarrow_handlers,
    *xarray_handlers,
    *scipy_handlers,
    *pillow_handlers,
    *networkx_handlers,
    *sklearn_handlers,
    *torch_handlers,
]

optional_handlers_by_type: HandlerTypeRegistry = {
    **msgspec_handlers_by_type,
    **attrs_handlers_by_type,
    **numpy_handlers_by_type,
    **pandas_handlers_by_type,
    **pydantic_handlers_by_type,
    **polars_handlers_by_type,
    **pyarrow_handlers_by_type,
    **xarray_handlers_by_type,
    **scipy_handlers_by_type,
    **pillow_handlers_by_type,
    **networkx_handlers_by_type,
    **sklearn_handlers_by_type,
    **torch_handlers_by_type,
}
