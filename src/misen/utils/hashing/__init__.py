"""Canonical object hashing with explicit type handlers.

Design notes:

- Hashes are type-aware: runtime type name is included in the top-level digest.
- Per-handler version is included so future implementation changes produce new hashes.
- Built-in handlers cover common declarative primitives/collections deterministically.
- Optional library handlers are matched dynamically when installed.
- Unknown types fail immediately.
"""

from functools import partial
from typing import Any

from misen.utils.hashing.handler_base import (
    Handler,
    HandlerTypeList,
    HandlerTypeRegistry,
    PrimitiveHandler,
    hash_values,
    qualified_type_name,
)
from misen.utils.hashing.handlers import (
    optional_handlers,
    optional_handlers_by_type,
    stdlib_handlers,
    stdlib_handlers_by_type,
)
from misen.utils.hashing.hash_types import Hash, ResolvedTaskHash, ResultHash, TaskHash

__all__ = [
    "Hash",
    "ResolvedTaskHash",
    "ResultHash",
    "TaskHash",
    "UnhashableTypeError",
    "get_handler_versions",
    "stable_hash",
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def stable_hash(
    obj: Any,
    *,
    handler_versions: dict[str, int] | None = None,
    _active_ids: set[int] | None = None,
) -> int:
    """Return a stable 64-bit hash for an arbitrary Python object.

    Parameters
    ----------
    handler_versions:
        Optional mapping of ``{qualified_type_name: version}`` pinning handler
        versions.  The workspace records this at creation time so that future
        handler upgrades can dispatch to old digest implementations.  For
        now, if a pinned version differs from the current handler version a
        ``ValueError`` is raised (version dispatch will be added when the
        first handler actually changes).
    """
    if _active_ids is None:
        _active_ids = set()

    handler_cls = _lookup_handler(obj)
    obj_type = handler_cls.type_name(obj)

    # Check pinned version against current handler version.
    if handler_versions is not None:
        pinned = handler_versions.get(obj_type)
        if pinned is not None and pinned != handler_cls.version:
            msg = (
                f"Handler for {obj_type} is at version {handler_cls.version}, "
                f"but workspace pins version {pinned}. Version dispatch is not "
                f"yet implemented — update the handler or migrate the workspace."
            )
            raise ValueError(msg)

    if issubclass(handler_cls, PrimitiveHandler):
        obj_hash = handler_cls.digest(obj)
    else:
        obj_id = id(obj)
        if obj_id in _active_ids:
            # Use a stable marker for back-edges to avoid infinite recursion.
            return hash_values((handler_cls.version, obj_type, "__recursive_reference__"))

        _active_ids.add(obj_id)
        try:
            element_hash = partial(stable_hash, handler_versions=handler_versions, _active_ids=_active_ids)
            obj_hash = handler_cls.digest(obj, element_hash=element_hash)
        finally:
            _active_ids.remove(obj_id)

    return hash_values((handler_cls.version, obj_type, obj_hash))


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------


class UnhashableTypeError(TypeError):
    """Raised when ``stable_hash`` is asked to hash a type without an explicit handler."""


_handlers_by_type_name: HandlerTypeRegistry = {**stdlib_handlers_by_type, **optional_handlers_by_type}
_handlers_type_cache: dict[type[Any], type[Handler]] = {}
_handler_types: HandlerTypeList = [*stdlib_handlers, *optional_handlers]


def _lookup_handler(obj: Any) -> type[Handler]:
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


def get_handler_versions() -> dict[str, int]:
    """Return ``{qualified_type_name: handler_version}`` for all registered handlers.

    The workspace should call this at creation time and persist the result.
    On subsequent loads, pass it to ``stable_hash(handler_versions=...)`` so
    that version mismatches are detected.
    """
    versions: dict[str, int] = {}
    for type_name, handler_cls in _handlers_by_type_name.items():
        versions[type_name] = handler_cls.version
    for handler_cls in _handler_types:
        name = qualified_type_name(handler_cls)
        if name not in versions:
            versions[name] = handler_cls.version
    return versions
