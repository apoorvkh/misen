"""Aggregated registry of all :class:`Handler` subclasses in ``libs/``.

Each submodule under ``libs/`` defines a ``<lib>_handlers`` list and a
``<lib>_handlers_by_type`` dict.  This module concatenates them into
``all_handlers`` (ordered by dispatch priority) and
``all_handlers_by_type`` (keyed by exact qualified type name) for
:mod:`misen.utils.hashing` to consume.

Adding a new handler library: create ``libs/<name>.py`` exposing
``<name>_handlers`` and ``<name>_handlers_by_type``, then import them
here.  Order in ``all_handlers`` determines priority for the
:meth:`Handler.match`-based linear-scan fallback; ``stdlib_handlers``
ships last because its primitives match the broadest set of types.
"""

from misen.utils.hashing.base import Handler, HandlerTypeRegistry
from misen.utils.hashing.libs.attrs import attrs_handlers, attrs_handlers_by_type
from misen.utils.hashing.libs.msgspec import msgspec_handlers, msgspec_handlers_by_type
from misen.utils.hashing.libs.numpy import numpy_handlers, numpy_handlers_by_type
from misen.utils.hashing.libs.pyarrow import pyarrow_handlers, pyarrow_handlers_by_type
from misen.utils.hashing.libs.pydantic import pydantic_handlers, pydantic_handlers_by_type
from misen.utils.hashing.libs.stdlib import stdlib_handlers, stdlib_handlers_by_type
from misen.utils.hashing.libs.sympy import sympy_handlers, sympy_handlers_by_type
from misen.utils.hashing.libs.torch import torch_handlers, torch_handlers_by_type

__all__ = ["all_handlers", "all_handlers_by_type"]

all_handlers: list[type[Handler]] = [
    *msgspec_handlers,
    *attrs_handlers,
    *numpy_handlers,
    *pyarrow_handlers,
    *pydantic_handlers,
    *sympy_handlers,
    *torch_handlers,
    *stdlib_handlers,
]

all_handlers_by_type: HandlerTypeRegistry = {
    **msgspec_handlers_by_type,
    **attrs_handlers_by_type,
    **numpy_handlers_by_type,
    **pyarrow_handlers_by_type,
    **pydantic_handlers_by_type,
    **sympy_handlers_by_type,
    **torch_handlers_by_type,
    **stdlib_handlers_by_type,
}
