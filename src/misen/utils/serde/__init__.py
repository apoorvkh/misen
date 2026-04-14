"""Type-aware object serialization with automatic dispatch.

Public API:

- :func:`save` — serialize a value into a directory. Type-driven:
  picks the right :class:`Serializer` for the value via the internal
  type registry, or uses the caller-provided ``ser_cls``.
- :func:`load` — deserialize whatever is in a directory. Content-driven:
  reads ``serde_meta.json`` and dispatches to whichever serializer wrote
  the directory, so the caller never needs the original value's type.
  An explicit ``ser_cls`` may be passed to override the on-disk record.
- :class:`Serializer` — base class for implementing new serializers by
  overriding ``write`` / ``read`` (and optionally ``match`` for
  participation in auto-dispatch).
- :class:`UnserializableTypeError` — raised by individual serializers
  when they cannot encode a given value.

Design notes:

- Each :class:`Serializer` targets specific Python types and uses a
  stable on-disk format. ``write`` and ``read`` are the only hooks a
  subclass must implement; the module-level ``save`` / ``load`` handle
  ``serde_meta.json`` uniformly so subclasses never touch it.
- Version information is not a first-class concept. A serializer that
  needs to discriminate between on-disk formats can record a
  ``format_version`` (or similar) in the extras dict returned by
  :meth:`Serializer.write` and branch on ``meta.get("format_version")``
  in :meth:`Serializer.read`.
- Unknown types raise ``TypeError`` from :func:`save` rather than
  falling back to an unstable format like ``dill``. To serialize a
  value of an unregistered type, pass an explicit ``ser_cls`` or
  convert the value first.
"""

from misen.utils.serde.base import (
    Serializer,
    SerializerTypeRegistry,
    UnserializableTypeError,
)
from misen.utils.serde.registry import load, save

__all__ = [
    "Serializer",
    "SerializerTypeRegistry",
    "UnserializableTypeError",
    "load",
    "save",
]
