"""Canonical object hashing with explicit type handlers.

Design notes:

- Hashes are type-aware: runtime type name is included in the top-level digest.
- Built-in handlers cover common declarative primitives/collections deterministically.
- Optional library handlers are matched dynamically when installed.
- Unknown types fail immediately.

Hash output is a persisted invariant — see ``base.py`` and
``tests/test_hash_pinned.py`` for the contract.
"""

from functools import partial
from typing import Any

from misen.exceptions import HashError
from misen.utils.hashing.base import Handler, hash_values, qualified_type_name
from misen.utils.hashing.hash_types import Hash, ResolvedTaskHash, ResultHash, TaskHash
from misen.utils.hashing.libs import all_handlers, all_handlers_by_type
from misen.utils.type_registry import TypeDispatchRegistry

__all__ = [
    "Hash",
    "ResolvedTaskHash",
    "ResultHash",
    "TaskHash",
    "stable_hash",
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def stable_hash(obj: Any, *, _active_ids: set[int] | None = None) -> int:
    """Return a stable 64-bit hash for an arbitrary Python object."""
    if _active_ids is None:
        _active_ids = set()

    handler_cls = _handler_registry.lookup(obj)
    if handler_cls is None:
        msg = (
            f"stable_hash does not support values of type {qualified_type_name(type(obj))}. "
            "stable_hash only hashes values with explicit handlers. Register a stable_hash handler "
            "or convert this value to a stable declarative identifier (for example a string, enum, "
            "or Literal value)."
        )
        raise HashError(msg)
    obj_type = handler_cls.type_name(obj)

    obj_id = id(obj)
    if obj_id in _active_ids:
        # Use a stable marker for back-edges to avoid infinite recursion.
        return hash_values((obj_type, "__recursive_reference__"))

    _active_ids.add(obj_id)
    try:
        element_hash = partial(stable_hash, _active_ids=_active_ids)
        obj_hash = handler_cls.digest(obj, element_hash)
    finally:
        _active_ids.remove(obj_id)

    return hash_values((obj_type, obj_hash))


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------


_handler_registry: TypeDispatchRegistry[type[Handler]] = TypeDispatchRegistry(
    by_type_name=all_handlers_by_type,
    candidates=all_handlers,
    predicate=lambda handler_cls, obj: handler_cls.match(obj),
)
