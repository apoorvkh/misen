"""Base :class:`Serializer` class and associated type aliases.

Kept free of registry and ``libs/`` dependencies so individual
serializer modules under ``libs/`` can subclass :class:`Serializer`
without triggering imports of every serializer in the project.

The class itself exposes only the ``write`` / ``read`` / ``match``
hooks for subclasses to implement. The public ``save`` / ``load``
entry points — which handle ``serde_meta.json`` and auto-dispatch —
live in :mod:`misen.utils.serde.registry` and are re-exported from
:mod:`misen.utils.serde`.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Generic, TypeAlias, TypeVar

__all__ = ["Serializer", "SerializerTypeRegistry", "UnserializableTypeError"]

T = TypeVar("T")


class UnserializableTypeError(TypeError):
    """Raised when a serializer is asked to encode a value it cannot represent.

    This wraps the underlying encoder error (typically ``TypeError`` or
    ``OverflowError`` from ``msgspec``) with a clear, domain-specific message
    that names the offending value's type.
    """


class Serializer(ABC, Generic[T]):
    """Serialize/deserialize an object to/from a directory.

    Subclasses implement three hooks:

    - :meth:`write` — write data files into ``directory``. May return a
      mapping of extra fields to record in ``serde_meta.json`` alongside
      the serializer name (e.g. a ``format_version`` to branch on when
      loading), or ``None`` if no extras are needed.
    - :meth:`read` — read data files from ``directory`` and reconstruct
      the object, given the full ``serde_meta.json`` contents.
    - :meth:`match` (optional) — return ``True`` if this serializer can
      handle a given object, used by :func:`misen.utils.serde.save` to
      pick a serializer when no explicit class is provided.

    Subclasses never call :func:`misen.utils.serde.save` or read/write
    ``serde_meta.json`` themselves — that is the module-level save/load's
    job. The class does not expose ``save``/``load`` directly.
    """

    @staticmethod
    def match(obj: Any) -> bool:  # noqa: ARG004
        """Return whether this serializer can handle *obj*.

        Consulted by :func:`misen.utils.serde.save` when no explicit
        ``ser_cls`` is provided and the type-name lookup does not find
        a match. The default returns ``False``, so user-defined
        serializers that only implement :meth:`write`/:meth:`read` are
        never selected by auto-dispatch — callers must opt in by
        passing ``ser_cls=...`` or by overriding :meth:`match`.
        """
        return False

    @staticmethod
    @abstractmethod
    def write(obj: T, directory: Path) -> Mapping[str, Any] | None:
        """Write data files for *obj* into *directory*.

        Return a mapping of extra fields to record in ``serde_meta.json``
        alongside the serializer name, or ``None`` if no extras are
        needed. Do not write ``serde_meta.json`` here —
        :func:`misen.utils.serde.save` handles that after this returns.
        """

    @staticmethod
    @abstractmethod
    def read(directory: Path, *, meta: Mapping[str, Any]) -> T:
        """Read data files from *directory* and reconstruct the object.

        *meta* is the full ``serde_meta.json`` contents, including the
        ``serializer`` name and any extras returned by :meth:`write`.
        Subclasses that discriminate between on-disk formats should
        record a version key from :meth:`write` and branch on it here.
        """


SerializerTypeRegistry: TypeAlias = dict[str, type[Serializer]]
