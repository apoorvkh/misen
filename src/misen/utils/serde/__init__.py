"""Type-aware object serialization with automatic dispatch.

Public API:

- :func:`save` — serialize a value into a directory.  Type-driven:
  picks the right :class:`Serializer` (or subclass) for the value via
  the internal type registry, or uses the caller-provided ``ser_cls``.
- :func:`load` — deserialize whatever is in a directory.  Content-driven:
  reads ``manifest.json`` and dispatches to whichever serializer wrote
  the directory, so the caller never needs the original value's type.
  An explicit ``ser_cls`` may be passed to override the on-disk record.
- :func:`save_zip` / :func:`load_zip` — single-file zip variants.

Subclassing for custom types:

- :class:`Serializer` — **start here**.  Override :meth:`~BaseSerializer.write`
  and :meth:`~BaseSerializer.read` to persist an object into a directory.
- :class:`LeafSerializer` — advanced, for batching many instances of
  one kind (tensors, ndarrays) into a single file.
- :class:`BaseSerializer` — internal.  Subclass directly only when
  writing a recursion-aware container serializer.
"""

from misen.exceptions import SerializationError
from misen.utils.serde.base import (
    BaseSerializer,
    Container,
    DecodeCtx,
    DirectoryLeaf,
    EncodeCtx,
    Leaf,
    LeafSerializer,
    Node,
    Serializer,
)
from misen.utils.serde.registry import (
    MANIFEST_FILENAME,
    Registry,
    load,
    load_zip,
    save,
    save_zip,
)

__all__ = [
    "MANIFEST_FILENAME",
    "BaseSerializer",
    "Container",
    "DecodeCtx",
    "DirectoryLeaf",
    "EncodeCtx",
    "Leaf",
    "LeafSerializer",
    "Node",
    "Registry",
    "SerializationError",
    "Serializer",
    "load",
    "load_zip",
    "save",
    "save_zip",
]
