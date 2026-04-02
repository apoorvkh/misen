"""Aggregate optional handlers for declarative third-party value types."""

from misen_hash.handler_base import HandlerTypeList, HandlerTypeRegistry
from misen_hash.handlers.optional.attrs import attrs_handlers, attrs_handlers_by_type
from misen_hash.handlers.optional.msgspec import msgspec_handlers, msgspec_handlers_by_type
from misen_hash.handlers.optional.nltk import nltk_handlers, nltk_handlers_by_type
from misen_hash.handlers.optional.numpy import numpy_handlers, numpy_handlers_by_type
from misen_hash.handlers.optional.pyarrow import pyarrow_handlers, pyarrow_handlers_by_type
from misen_hash.handlers.optional.pydantic import pydantic_handlers, pydantic_handlers_by_type
from misen_hash.handlers.optional.sympy import sympy_handlers, sympy_handlers_by_type
from misen_hash.handlers.optional.transformers import transformers_handlers, transformers_handlers_by_type

__all__ = ["optional_handlers", "optional_handlers_by_type"]

optional_handlers: HandlerTypeList = [
    *msgspec_handlers,
    *attrs_handlers,
    *nltk_handlers,
    *numpy_handlers,
    *pydantic_handlers,
    *pyarrow_handlers,
    *sympy_handlers,
    *transformers_handlers,
]

optional_handlers_by_type: HandlerTypeRegistry = {
    **msgspec_handlers_by_type,
    **attrs_handlers_by_type,
    **nltk_handlers_by_type,
    **numpy_handlers_by_type,
    **pydantic_handlers_by_type,
    **pyarrow_handlers_by_type,
    **sympy_handlers_by_type,
    **transformers_handlers_by_type,
}
