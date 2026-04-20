"""Module-level ``save`` / ``load`` entry points.

Ties together the type-based dispatch registry built from
:mod:`misen.utils.serde.libs` with the on-disk ``serde_meta.json``
format. :func:`save` picks a :class:`Serializer` for a value (or uses
the one the caller passes), delegates file writes to its ``write``
hook, then records the serializer's qualified name in
``serde_meta.json``. :func:`load` reverses the process: it reads the
metadata, looks the serializer back up by qualified name, and calls
its ``read`` hook.

Both functions are re-exported from :mod:`misen.utils.serde`; they are
the only public save/load surface the package exposes.
"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from misen.exceptions import SerializationError
from misen.utils.serde.base import Serializer
from misen.utils.serde.libs import all_serializers, all_serializers_by_type
from misen.utils.type_registry import TypeDispatchRegistry, qualified_type_name

__all__ = ["load", "save"]

_META_FILENAME = "serde_meta.json"


# ``dict`` / ``OrderedDict`` are content-sensitive: the same concrete type
# can dispatch to different serializers depending on value types (e.g.
# ``DictOfTensorsSerializer`` for dicts of ``torch.Tensor``,
# ``MsgpackSerializer`` for dicts of primitives).  Listing them as volatile
# bypasses both the cache and the by-type fast path so every call
# re-evaluates the candidate match predicates.
_serializer_registry: TypeDispatchRegistry[type[Serializer]] = TypeDispatchRegistry(
    by_type_name=all_serializers_by_type,
    candidates=all_serializers,
    predicate=lambda ser_cls, obj: ser_cls.match(obj),
    volatile_types={dict, OrderedDict},
)

# Map from serializer qualified name → class (for meta-based load dispatch).
_serializer_by_qualified_name: dict[str, type[Serializer]] = {
    qualified_type_name(ser_cls): ser_cls for ser_cls in _serializer_registry.candidates
}


def save(obj: Any, directory: Path, ser_cls: type[Serializer] | None = None) -> None:
    """Serialize *obj* into *directory*, writing ``serde_meta.json``.

    Dispatches to *ser_cls* if given; otherwise looks up the best
    serializer for *obj* in the type registry (by exact type name
    first, then by :meth:`Serializer.match` on remaining candidates).

    Args:
        obj: Value to serialize.
        directory: Existing directory to write data files into.
        ser_cls: Optional explicit serializer class; bypasses dispatch.

    Raises:
        SerializationError: If no serializer is registered for
            ``type(obj)`` and no *ser_cls* was provided.
    """
    ser_cls = ser_cls or _serializer_registry.lookup(obj)
    if ser_cls is None:
        msg = (
            f"No serializer registered for type {qualified_type_name(type(obj))!r}. "
            "Either pass a custom serializer to @meta(serializer=...) or convert "
            "the return value to a supported type."
        )
        raise SerializationError(msg)
    extra = ser_cls.write(obj, directory) or {}

    # Write ``serde_meta.json`` recording the serializer used
    meta = {"serializer": qualified_type_name(ser_cls), **extra}
    (directory / _META_FILENAME).write_text(json.dumps(meta), encoding="utf-8")


def load(directory: Path, ser_cls: type[Serializer] | None = None) -> Any:
    """Deserialize the object stored in *directory*.

    Reads ``serde_meta.json`` and, unless *ser_cls* is explicitly
    provided, looks the serializer back up by the qualified name
    recorded there. Loading is normally content-driven — the metadata
    is authoritative — so *ser_cls* is only needed when the caller
    knows better than the on-disk record (e.g. the original class has
    been renamed but remains wire-compatible).

    Args:
        directory: Directory written by a previous :func:`save` call.
        ser_cls: Optional explicit serializer class; overrides the
            class named in ``serde_meta.json``.

    Raises:
        SerializationError: If ``serde_meta.json`` is missing,
            malformed, or names a serializer that is no longer
            registered and no *ser_cls* was provided.
    """
    try:
        meta: dict[str, Any] = json.loads((directory / _META_FILENAME).read_text(encoding="utf-8"))
    except FileNotFoundError:
        msg = f"No serde_meta.json found in {directory}"
        raise SerializationError(msg) from None

    if "serializer" not in meta:
        msg = f"serde_meta.json in {directory} does not contain a 'serializer' field"
        raise SerializationError(msg)

    ser_name = meta["serializer"]
    ser_cls = ser_cls or _serializer_by_qualified_name.get(ser_name)
    if ser_cls is None:
        msg = f"Unknown serializer {ser_name!r} in serde_meta.json. The serializer may have been renamed or removed."
        raise SerializationError(msg)
    return ser_cls.read(directory, meta=meta)
