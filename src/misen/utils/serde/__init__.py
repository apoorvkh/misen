"""Type-aware object serialization with automatic dispatch.

Design notes:

- Each serializer targets specific Python types and uses a stable on-disk format.
- ``DefaultSerializer`` auto-selects the best serializer via a type registry and
  ``match()`` fallback scan (same algorithm as ``misen.utils.hashing``).
- Unknown types raise ``TypeError`` rather than falling back to an unstable
  format like ``dill``.
- Metadata (``serde_meta.json``) records which serializer wrote the data so
  that ``load`` can dispatch without the original object.
"""

import warnings
from pathlib import Path
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerClass,
    SerializerTypeList,
    SerializerTypeRegistry,
    qualified_type_name,
    read_meta,
)
from misen.utils.serde.serializers import all_serializers, all_serializers_by_type

__all__ = ["DefaultSerializer", "Serializer"]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_serializers_by_type_name: SerializerTypeRegistry = dict(all_serializers_by_type)
_serializer_type_cache: dict[type[Any], SerializerClass] = {}
_serializer_list: SerializerTypeList = list(all_serializers)

# Map from serializer qualified name → class (for meta-based load dispatch).
_serializer_by_qualified_name: dict[str, SerializerClass] = {}
for _ser_cls in _serializer_list:
    _serializer_by_qualified_name[qualified_type_name(_ser_cls)] = _ser_cls


def _lookup_serializer(obj: Any) -> SerializerClass:
    """Resolve the best serializer class for *obj*, memoized by runtime type."""
    obj_type = type(obj)

    cached = _serializer_type_cache.get(obj_type)
    if cached is not None:
        return cached

    # Fast path: check type name registry (exact type and MRO bases).
    for base_type in obj_type.__mro__:
        by_name = _serializers_by_type_name.get(qualified_type_name(base_type))
        if by_name is not None:
            _serializer_type_cache[obj_type] = by_name
            return by_name

    # Slow path: linear scan calling match().
    for ser_cls in _serializer_list:
        if ser_cls.match(obj):
            _serializer_type_cache[obj_type] = ser_cls
            return ser_cls

    msg = (
        f"No serializer registered for type {qualified_type_name(obj_type)!r}. "
        "Either pass a custom serializer to @task(serializer=...) or convert "
        "the return value to a supported type."
    )
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# DefaultSerializer
# ---------------------------------------------------------------------------


class DefaultSerializer(Serializer[Any]):
    """Auto-dispatching serializer that selects the best format per type."""

    __slots__ = ()

    @staticmethod
    def save(obj: Any, directory: Path) -> None:
        """Serialize *obj* using the best available type-specific serializer."""
        ser_cls = _lookup_serializer(obj)
        ser_cls.save(obj, directory)

    @staticmethod
    def load(directory: Path) -> Any:
        """Deserialize from *directory*, using metadata to select the serializer."""
        meta = read_meta(directory)

        if meta is None:
            msg = f"No serde_meta.json found in {directory}"
            raise ValueError(msg)

        ser_name = meta["serializer"]
        ser_cls = _serializer_by_qualified_name.get(ser_name)
        if ser_cls is None:
            msg = (
                f"Unknown serializer {ser_name!r} in serde_meta.json. "
                "The serializer may have been renamed or removed."
            )
            raise ValueError(msg)

        saved_version = meta.get("version")
        if saved_version is not None and saved_version != ser_cls.version:
            warnings.warn(
                f"Serializer {ser_name} was saved at version {saved_version} "
                f"but the current version is {ser_cls.version}. "
                f"The data may not load correctly.",
                stacklevel=2,
            )

        return ser_cls.load(directory)
