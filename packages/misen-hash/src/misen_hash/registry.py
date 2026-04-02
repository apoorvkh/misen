"""Handler registry assembly and handler resolution."""

from typing import Any

from misen_hash.handler_base import Handler, HandlerTypeList, HandlerTypeRegistry, qualified_type_name
from misen_hash.handlers import (
    builtin_handlers,
    builtin_handlers_by_type,
    optional_handlers,
    optional_handlers_by_type,
)

__all__ = ["UnhashableTypeError", "lookup_handler"]


class UnhashableTypeError(TypeError):
    """Raised when ``stable_hash`` is asked to hash a type without an explicit handler."""


_handlers_by_type_name: HandlerTypeRegistry = {**builtin_handlers_by_type, **optional_handlers_by_type}
_handlers_type_cache: dict[type[Any], type[Handler]] = {}
_handler_types: HandlerTypeList = [*builtin_handlers, *optional_handlers]


def lookup_handler(obj: Any) -> type[Handler]:
    """Resolve the most specific handler class for ``obj`` and memoize by type."""
    obj_type = type(obj)

    cached = _handlers_type_cache.get(obj_type)
    if cached is not None:
        return cached

    for base_type in obj_type.__mro__:
        by_name = _handlers_by_type_name.get(qualified_type_name(base_type))
        if by_name is not None:
            _handlers_type_cache[obj_type] = by_name
            return by_name

    for hash_cls in _handler_types:
        if hash_cls.match(obj):
            _handlers_type_cache[obj_type] = hash_cls
            return hash_cls

    msg = (
        f"stable_hash does not support values of type {qualified_type_name(obj_type)}. "
        "stable_hash only hashes values with explicit handlers. Register a stable_hash handler "
        "or convert this value to a stable declarative identifier (for example a string, enum, "
        "or Literal value)."
    )
    raise UnhashableTypeError(msg)
