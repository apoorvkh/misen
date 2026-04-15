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
        Optional mapping of ``{handler_qualified_name: version}`` pinning
        handler versions.  Keyed on the handler class's qualified name
        (e.g. ``"misen.utils.hashing.libs.stdlib.IntHandler"``) so that
        match-only handlers — whose target types are open-ended — are
        versioned by the handler rather than the hashed value's type.

        The workspace records the current versions at creation time via
        :func:`get_handler_versions` and passes them back here on
        subsequent runs.  If a pinned version differs from the current
        handler version a ``ValueError`` is raised (version dispatch
        will be added when the first handler actually changes).
    """
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

    # Check pinned version against current handler version — keyed on the
    # handler class so match-only handlers (attrs, pydantic, …) are covered.
    if handler_versions is not None:
        handler_qname = qualified_type_name(handler_cls)
        pinned = handler_versions.get(handler_qname)
        if pinned is not None and pinned != handler_cls.version:
            msg = (
                f"Handler {handler_qname} is at version {handler_cls.version}, "
                f"but workspace pins version {pinned}. Version dispatch is not "
                f"yet implemented — update the handler or migrate the workspace."
            )
            raise ValueError(msg)

    obj_id = id(obj)
    if obj_id in _active_ids:
        # Use a stable marker for back-edges to avoid infinite recursion.
        return hash_values((handler_cls.version, obj_type, "__recursive_reference__"))

    _active_ids.add(obj_id)
    try:
        element_hash = partial(stable_hash, handler_versions=handler_versions, _active_ids=_active_ids)
        obj_hash = handler_cls.digest(obj, element_hash)
    finally:
        _active_ids.remove(obj_id)

    return hash_values((handler_cls.version, obj_type, obj_hash))


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------


_handler_registry: TypeDispatchRegistry[type[Handler]] = TypeDispatchRegistry(
    by_type_name=all_handlers_by_type,
    candidates=all_handlers,
    predicate=lambda handler_cls, obj: handler_cls.match(obj),
)


def get_handler_versions() -> dict[str, int]:
    """Return ``{handler_qualified_name: handler_version}`` for all registered handlers.

    Keyed on the handler class's qualified name (not the target type) so
    that each registered handler appears exactly once and match-only
    handlers (:class:`AttrsHandler`, :class:`PydanticModelHandler`, …)
    are included.  Multiple target types sharing a handler (e.g. ``list``,
    ``tuple``, ``set``, ``frozenset`` all using ``ListHandler``) collapse
    to a single entry.

    The workspace should call this at creation time and persist the
    result.  On subsequent loads, pass it to
    ``stable_hash(handler_versions=...)`` so that version mismatches
    are detected.
    """
    # Deduplicate handlers seen via by_type_name + linear-scan candidates.
    seen: dict[str, int] = {}
    for handler_cls in (*_handler_registry.by_type_name.values(), *_handler_registry.candidates):
        seen[qualified_type_name(handler_cls)] = handler_cls.version
    return seen
