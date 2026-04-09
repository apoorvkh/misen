"""Base serializer abstraction and metadata helpers."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

__all__ = [
    "Serializer",
    "SerializerClass",
    "SerializerTypeList",
    "SerializerTypeRegistry",
    "qualified_type_name",
    "read_meta",
    "write_meta",
]

T = TypeVar("T")

_META_FILENAME = "serde_meta.json"


def qualified_type_name(obj_type: type[Any]) -> str:
    """Return the fully-qualified ``module.qualname`` for *obj_type*."""
    return f"{obj_type.__module__}.{obj_type.__qualname__}"


class Serializer(ABC, Generic[T]):
    """Serialize/deserialize an object to/from a directory.

    Subclass and override :meth:`save` and :meth:`load` to implement a
    type-specific serializer.  Optionally override :meth:`match` so that
    :class:`DefaultSerializer` can auto-select the serializer for a given
    object.

    The ``version`` class variable is recorded in metadata and can be used
    to dispatch to legacy load logic when the on-disk format changes.
    """

    __slots__ = ()

    version: int = 1

    @staticmethod
    def match(obj: Any) -> bool:  # noqa: ARG004
        """Return whether this serializer can handle *obj*.

        The default returns ``False`` so that user-defined serializers that
        only implement :meth:`save`/:meth:`load` are never selected by the
        automatic dispatch in :class:`DefaultSerializer`.
        """
        return False

    @staticmethod
    @abstractmethod
    def save(obj: T, directory: Path) -> None:
        """Serialize *obj* into *directory*."""

    @staticmethod
    @abstractmethod
    def load(directory: Path) -> T:
        """Deserialize and return the object stored in *directory*."""


# ---------------------------------------------------------------------------
# Type aliases (mirrors hashing handler_base)
# ---------------------------------------------------------------------------

SerializerClass = type[Serializer]
SerializerTypeList = list[SerializerClass]
SerializerTypeRegistry = dict[str, SerializerClass]


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def write_meta(directory: Path, serializer_cls: SerializerClass, **extra: Any) -> None:
    """Write ``serde_meta.json`` recording the serializer used."""
    meta = {
        "serializer": qualified_type_name(serializer_cls),
        "version": serializer_cls.version,
        **extra,
    }
    (directory / _META_FILENAME).write_text(json.dumps(meta), encoding="utf-8")


def read_meta(directory: Path) -> dict[str, Any] | None:
    """Read ``serde_meta.json``, returning ``None`` if absent."""
    path = directory / _META_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
