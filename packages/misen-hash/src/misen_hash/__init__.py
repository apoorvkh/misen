"""Canonical object hashing with extensible type handlers.

Design notes:

- Hashes are type-aware: runtime type name is included in the top-level digest.
- Built-in handlers cover common primitives/collections deterministically.
- Optional library handlers are matched dynamically when installed.
- Unknown types fall back to dill serialization.
"""

from functools import partial
from typing import Any

from misen_hash.handler_base import PrimitiveHandler, qualified_type_name
from misen_hash.hash import hash_msgspec
from misen_hash.registry import lookup_handler

__all__ = ["stable_hash"]


def stable_hash(obj: Any, *, _active_ids: set[int] | None = None) -> int:
    """Return a stable 64-bit hash for an arbitrary Python object."""
    if _active_ids is None:
        _active_ids = set()

    obj_type = qualified_type_name(type(obj))
    handler_cls = lookup_handler(obj)

    if issubclass(handler_cls, PrimitiveHandler):
        obj_hash = handler_cls.digest(obj)
    else:
        obj_id = id(obj)
        if obj_id in _active_ids:
            # Use a stable marker for back-edges to avoid infinite recursion.
            return hash_msgspec((obj_type, "__recursive_reference__"))

        _active_ids.add(obj_id)
        try:
            obj_hash = handler_cls.digest(obj, element_hash=partial(stable_hash, _active_ids=_active_ids))
        finally:
            _active_ids.remove(obj_id)

    return hash_msgspec((obj_type, obj_hash))
