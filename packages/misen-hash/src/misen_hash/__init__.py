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

from misen_hash.handler_base import PrimitiveHandler
from misen_hash.hash import canonical_hash
from misen_hash.registry import UnhashableTypeError, get_handler_versions, lookup_handler

__all__ = ["UnhashableTypeError", "get_handler_versions", "stable_hash"]


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
        misen-hash upgrades can dispatch to old digest implementations.  For
        now, if a pinned version differs from the current handler version a
        ``ValueError`` is raised (version dispatch will be added when the
        first handler actually changes).
    """
    if _active_ids is None:
        _active_ids = set()

    handler_cls = lookup_handler(obj)
    obj_type = handler_cls.type_name(obj)

    # Check pinned version against current handler version.
    if handler_versions is not None:
        pinned = handler_versions.get(obj_type)
        if pinned is not None and pinned != handler_cls.version:
            msg = (
                f"Handler for {obj_type} is at version {handler_cls.version}, "
                f"but workspace pins version {pinned}. Version dispatch is not "
                f"yet implemented — update misen-hash or migrate the workspace."
            )
            raise ValueError(msg)

    if issubclass(handler_cls, PrimitiveHandler):
        obj_hash = handler_cls.digest(obj)
    else:
        obj_id = id(obj)
        if obj_id in _active_ids:
            # Use a stable marker for back-edges to avoid infinite recursion.
            return canonical_hash((handler_cls.version, obj_type, "__recursive_reference__"))

        _active_ids.add(obj_id)
        try:
            element_hash = partial(stable_hash, handler_versions=handler_versions, _active_ids=_active_ids)
            obj_hash = handler_cls.digest(obj, element_hash=element_hash)
        finally:
            _active_ids.remove(obj_id)

    return canonical_hash((handler_cls.version, obj_type, obj_hash))
