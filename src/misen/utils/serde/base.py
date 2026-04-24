"""Node types, serializer base classes, and encode/decode contexts.

The design splits a serializer's job into two phases:

1. **Encode** — walks the object and returns a :class:`Node` graph.
   Leaves hold their payloads in the :class:`EncodeCtx`; containers
   recurse on their children, with :class:`Ref` nodes for repeated
   non-leaf objects.
2. **Commit** — after the walk, :func:`~misen.utils.serde.save` groups
   leaves by ``leaf_kind`` and asks each owning serializer to write
   the whole group in a single batched call (e.g. one ``torch.save``
   of all tensors in the graph).  A single ``manifest.json`` records
   the graph.

Self-referential (cyclic) graphs only round-trip through mutable
containers whose serializer publishes a placeholder via
:meth:`DecodeCtx.remember_node` before decoding children (built-in
``dict`` and ``list`` do this).  Serializers that must fully construct
the object from decoded children — dataclasses, pydantic models,
attrs, msgspec Structs, tuples — cannot resolve a cycle back to
themselves because the constructor needs values that don't yet exist.

Serializer shapes, by user audience:

- :class:`Serializer` — **most users want this**.  Subclass and
  implement :meth:`~Serializer.write` / :meth:`~Serializer.read` to
  persist an object into a directory.  This matches the classic "save
  files here, read them back" mental model.
- :class:`LeafSerializer` — advanced.  For types where saving many
  instances together in one file is a real win (tensors, ndarrays).
- :class:`BaseSerializer` — internal.  Subclass directly only when
  writing a recursion-aware container serializer (e.g. a custom
  mapping type whose children should dispatch independently).  The
  framework's ``DictSerializer`` and ``ListSerializer`` in
  ``libs/stdlib.py`` are examples.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

from misen.exceptions import SerializationError
from misen.utils.type_registry import qualified_type_name

__all__ = [
    "BaseSerializer",
    "Container",
    "DecodeCtx",
    "DirectoryLeaf",
    "EncodeCtx",
    "Leaf",
    "LeafSerializer",
    "Node",
    "Ref",
    "Serializer",
]

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Leaf:
    """A terminal node.  Its payload is batched with other leaves of the same kind.

    The payload itself lives in :class:`EncodeCtx` — the :class:`Leaf`
    only carries the ``leaf_id`` needed to retrieve it from the batched
    on-disk blob.  ``kind`` selects which serializer's
    ``write_batch``/``read_batch`` pair owns that blob.
    """

    serializer: str  # qualified name of the producing serializer class
    kind: str  # leaf kind, used to group/batch (e.g. "torch_tensor", "msgpack")
    leaf_id: str  # unique id within this save
    meta: Mapping[str, Any] = field(default_factory=dict)
    node_id: str | None = None


@dataclass(frozen=True)
class Container:
    """A structural node whose children are themselves :class:`Node` objects.

    ``children`` is either a :class:`Mapping` (for keyed containers like
    ``dict``) or a :class:`Sequence` (for indexed containers like
    ``list``).  The owning serializer decides the shape and reconstructs
    the correct Python type on decode.
    """

    serializer: str
    children: Any  # Mapping[str, Node] or Sequence[Node]
    meta: Mapping[str, Any] = field(default_factory=dict)
    node_id: str | None = None


@dataclass(frozen=True)
class DirectoryLeaf:
    """A serializer owns its own subdirectory.

    Produced by :class:`Serializer` (and any subclass that overrides
    :meth:`~Serializer.write` / :meth:`~Serializer.read`).
    """

    serializer: str
    subdir: str  # subdirectory name under ``dirs/`` in the root
    meta: Mapping[str, Any] = field(default_factory=dict)
    node_id: str | None = None


@dataclass(frozen=True)
class Ref:
    """A reference to an earlier node in the manifest graph.

    References let shared containers/directory leaves decode as shared
    Python objects.  They also let recursive mutable containers point
    back to placeholders registered during decode.
    Constructor-based serializers (tuple/dataclass/pydantic/etc.) cannot
    resolve self-cycles because their children must decode first.
    """

    ref_id: str


Node = Leaf | Container | DirectoryLeaf | Ref


# ---------------------------------------------------------------------------
# Serializer base classes
# ---------------------------------------------------------------------------


class BaseSerializer(ABC, Generic[T]):
    """Internal abstract base for all serializers.

    External users almost always want :class:`Serializer` (the
    user-facing ``write``/``read`` subclass) or :class:`LeafSerializer`
    (advanced batching).  Subclass :class:`BaseSerializer` directly
    only when writing a recursion-aware container — implement
    :meth:`encode` (producing a :class:`Container`) and :meth:`decode`.
    """

    # Leaf-kind owners set this to the kind their ``write_batch`` handles.
    leaf_kind: ClassVar[str | None] = None

    @staticmethod
    def match(obj: Any) -> bool:  # noqa: ARG004
        """Return ``True`` if this serializer can handle *obj*."""
        return False

    @classmethod
    @abstractmethod
    def encode(cls, obj: T, ctx: "EncodeCtx") -> Node:
        """Walk *obj* and return a :class:`Node` describing it."""

    @classmethod
    @abstractmethod
    def decode(cls, node: Node, ctx: "DecodeCtx") -> T:
        """Reconstruct the object described by *node*."""

    # ----- Leaf-kind owners override these -----

    @staticmethod
    def write_batch(
        entries: list[tuple[str, Any, Mapping[str, Any]]],
        directory: Path,
    ) -> Mapping[str, Any]:
        """Write all leaves of this kind to *directory* as one blob.

        ``entries`` is a list of ``(leaf_id, payload, per_leaf_meta)``.
        Return a mapping of kind-scoped metadata to record in the
        manifest (passed back to :meth:`read_batch`).
        """
        raise NotImplementedError

    @staticmethod
    def read_batch(
        directory: Path,
        kind_meta: Mapping[str, Any],
    ) -> Callable[[str], Any]:
        """Open the batched blob and return a ``leaf_id → payload`` reader."""
        raise NotImplementedError

    # ----- Directory-owning serializers override these -----

    @staticmethod
    def write(obj: Any, directory: Path) -> Mapping[str, Any] | None:
        """Write *obj*'s files into *directory*.

        Return a mapping of metadata to record alongside this
        directory in the manifest (passed back to :meth:`read` as
        ``meta``), or ``None`` if no extras are needed.
        """
        raise NotImplementedError

    @staticmethod
    def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
        """Reconstruct the object from files in *directory*.

        ``meta`` is whatever :meth:`write` returned for this directory.
        """
        raise NotImplementedError


class LeafSerializer(BaseSerializer[T]):
    """Advanced: batch many instances of one kind into a single blob.

    Subclasses set :attr:`leaf_kind` and implement
    :meth:`~BaseSerializer.write_batch` / :meth:`~BaseSerializer.read_batch`.
    Optionally override :meth:`to_payload` (to transform the object
    before batching — e.g. ``.detach().cpu()`` for tensors) or
    :meth:`to_meta` (for per-leaf structural metadata).

    Use only when batching is a real win (arrays/tensors).  For custom
    types, start with :class:`Serializer`.
    """

    # Subclasses MUST set this — the empty string acts as a sentinel
    # so a missing override fails loudly at dispatch time.
    leaf_kind: ClassVar[str] = ""

    @classmethod
    def to_payload(cls, obj: T) -> Any:
        """Return the value to batch — default is the object itself."""
        return obj

    @classmethod
    def to_meta(cls, obj: T) -> Mapping[str, Any]:  # noqa: ARG003
        """Return per-leaf metadata to record alongside the leaf id."""
        return {}

    @classmethod
    def encode(cls, obj: T, ctx: "EncodeCtx") -> Node:
        return ctx.add_leaf(cls, cls.leaf_kind, cls.to_payload(obj), cls.to_meta(obj))

    @classmethod
    def decode(cls, node: Node, ctx: "DecodeCtx") -> T:
        if not isinstance(node, Leaf):
            msg = f"{qualified_type_name(cls)} expected a Leaf node, got {type(node).__name__}."
            raise SerializationError(msg)
        return ctx.load_leaf(cls.leaf_kind, node.leaf_id)


class Serializer(BaseSerializer[T]):
    """User-facing default: save an object to a directory, read it back.

    Subclass and implement :meth:`~BaseSerializer.write` and
    :meth:`~BaseSerializer.read`.  The framework hands each call a
    fresh subdirectory and records whatever ``write`` returns in the
    manifest, passing it back to ``read`` as ``meta``.

    Example::

        class PickleSerializer(Serializer[Any]):
            @staticmethod
            def match(obj):
                return True  # catch-all for this demo

            @staticmethod
            def write(obj, directory):
                import pickle
                (directory / "data.pkl").write_bytes(pickle.dumps(obj))
                return None

            @staticmethod
            def read(directory, *, meta):
                import pickle
                return pickle.loads((directory / "data.pkl").read_bytes())
    """

    @classmethod
    def encode(cls, obj: T, ctx: "EncodeCtx") -> Node:
        return ctx.add_directory_leaf(cls, obj)

    @classmethod
    def decode(cls, node: Node, ctx: "DecodeCtx") -> T:
        if not isinstance(node, DirectoryLeaf):
            msg = f"{qualified_type_name(cls)} expected a DirectoryLeaf node, got {type(node).__name__}."
            raise SerializationError(msg)
        return ctx.read_directory_leaf(cls, node)


# ---------------------------------------------------------------------------
# Encode/decode contexts
# ---------------------------------------------------------------------------


class EncodeCtx:
    """State for one encode walk.

    Collects leaves and directory-leaves as the recursion proceeds.
    Memoizes by ``id(obj)`` so shared leaves reuse one ``leaf_id`` and
    shared non-leaf objects become :class:`Ref` nodes.  In-progress refs
    let mutable containers point back to themselves.
    """

    def __init__(self, registry: Any) -> None:
        self._registry = registry
        # Memo keyed on ``id(obj)`` — the user's reference to the root
        # object keeps every child alive for the whole walk, so no need
        # to retain a separate strong-ref list against id recycling.
        self._memo: dict[int, Node] = {}
        self._node_ids: dict[int, str] = {}
        self._in_progress: set[int] = set()
        self._leaves: dict[str, list[tuple[str, Any, Mapping[str, Any]]]] = {}
        self._leaf_owner: dict[str, type[BaseSerializer]] = {}
        self._dir_leaves: list[tuple[type[BaseSerializer], str, Any]] = []
        self._counter = 0
        self._node_counter = 0

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def _next_node_id(self) -> str:
        self._node_counter += 1
        return f"N{self._node_counter}"

    def encode(self, obj: Any, *, ser_cls: type[BaseSerializer] | None = None) -> Node:
        """Dispatch *obj* through the registry and return its :class:`Node`."""
        oid = id(obj)
        if oid in self._memo:
            node = self._memo[oid]
            # Leaf payloads already have identity caching by leaf_id, and
            # keeping repeated leaf nodes preserves the compact v1-style
            # manifests tests rely on.  Non-leaf repeated objects need refs
            # so decode can return the same Python object.
            if isinstance(node, Leaf):
                return node
            node_id = _node_id(node)
            if node_id is None:
                msg = f"Cannot reference memoized node without node_id: {node!r}"
                raise SerializationError(msg)
            return Ref(node_id)
        if oid in self._in_progress:
            return Ref(self._node_ids[oid])

        ser = ser_cls or self._registry.lookup(obj)
        if ser is None:
            msg = f"No serializer registered for type {qualified_type_name(type(obj))!r}."
            raise SerializationError(msg)
        node_id = self._next_node_id()
        self._node_ids[oid] = node_id
        self._in_progress.add(oid)
        try:
            node = ser.encode(obj, self)
        finally:
            self._in_progress.discard(oid)
        node = _with_node_id(node, node_id)
        self._memo[oid] = node
        return node

    def add_leaf(
        self,
        owner: type[BaseSerializer],
        kind: str,
        payload: Any,
        meta: Mapping[str, Any] | None = None,
    ) -> Leaf:
        """Register a leaf and return its :class:`Leaf` node."""
        if not kind:
            msg = f"{qualified_type_name(owner)} must set a non-empty leaf_kind."
            raise SerializationError(msg)
        existing_owner = self._leaf_owner.get(kind)
        if existing_owner is not None and existing_owner is not owner:
            msg = (
                f"Leaf kind {kind!r} is already owned by {qualified_type_name(existing_owner)}; "
                f"{qualified_type_name(owner)} cannot also write it."
            )
            raise SerializationError(msg)
        leaf_id = self._next_id("L")
        meta_dict = dict(meta or {})
        self._leaves.setdefault(kind, []).append((leaf_id, payload, meta_dict))
        self._leaf_owner[kind] = owner
        return Leaf(serializer=qualified_type_name(owner), kind=kind, leaf_id=leaf_id, meta=meta_dict)

    def add_directory_leaf(
        self,
        owner: type[BaseSerializer],
        payload: Any,
        meta: Mapping[str, Any] | None = None,
    ) -> DirectoryLeaf:
        """Register a directory-owning leaf and return its :class:`DirectoryLeaf`."""
        subdir = self._next_id("D")
        self._dir_leaves.append((owner, subdir, payload))
        return DirectoryLeaf(serializer=qualified_type_name(owner), subdir=subdir, meta=dict(meta or {}))

    # Accessors used by registry.save during the commit phase.

    @property
    def leaves_by_kind(self) -> Mapping[str, list[tuple[str, Any, Mapping[str, Any]]]]:
        return self._leaves

    @property
    def leaf_owners(self) -> Mapping[str, type[BaseSerializer]]:
        return self._leaf_owner

    @property
    def directory_leaves(self) -> list[tuple[type[BaseSerializer], str, Any]]:
        return self._dir_leaves


class DecodeCtx:
    """State for one decode walk.

    Caches leaf payloads by ``leaf_id`` so a shared leaf referenced
    twice in the graph decodes to the same Python object.  Caches
    concrete nodes by ``node_id`` so :class:`Ref` nodes can preserve
    shared containers and resolve recursive mutable containers.
    """

    def __init__(self, registry: Any, root_directory: Path, manifest: Mapping[str, Any]) -> None:
        self._registry = registry
        self._root = root_directory
        self._manifest = manifest
        self._leaf_readers: dict[str, Callable[[str], Any]] = {}
        self._leaf_cache: dict[str, Any] = {}
        self._node_cache: dict[str, Any] = {}

    def decode(self, node: Node) -> Any:
        """Reconstruct the object described by *node*."""
        if isinstance(node, Ref):
            try:
                return self._node_cache[node.ref_id]
            except KeyError:
                msg = f"Dangling or unsupported forward reference to node {node.ref_id!r}."
                raise SerializationError(msg) from None
        node_id = _node_id(node)
        if node_id is not None and node_id in self._node_cache:
            return self._node_cache[node_id]
        ser = self._registry.by_name(node.serializer)
        value = ser.decode(node, self)
        if node_id is not None:
            self._node_cache.setdefault(node_id, value)
        return value

    def remember_node(self, node: Node, value: Any) -> None:
        """Cache *value* for ``node.node_id`` before decoding its children."""
        node_id = _node_id(node)
        if node_id is not None:
            self._node_cache[node_id] = value

    def load_leaf(self, kind: str, leaf_id: str) -> Any:
        """Fetch a leaf payload, opening the batched blob lazily + caching."""
        if leaf_id in self._leaf_cache:
            return self._leaf_cache[leaf_id]
        reader = self._leaf_readers.get(kind)
        if reader is None:
            try:
                leaf_entry = self._manifest["leaves"][kind]
            except KeyError:
                msg = f"Manifest is missing leaf batch {kind!r} for leaf {leaf_id!r}."
                raise SerializationError(msg) from None
            owner_name = leaf_entry["serializer"]
            owner = self._registry.by_name(owner_name)
            kind_meta = leaf_entry.get("meta", {})
            blob_dir = self._root / "leaves" / kind
            reader = owner.read_batch(blob_dir, kind_meta)
            self._leaf_readers[kind] = reader
        value = reader(leaf_id)
        self._leaf_cache[leaf_id] = value
        return value

    def read_directory_leaf(self, owner: type[BaseSerializer], node: DirectoryLeaf) -> Any:
        """Invoke a directory-owning serializer on its subdir.

        Meta seen by :meth:`~BaseSerializer.read` merges (a) the
        :class:`DirectoryLeaf`'s node-level meta set at encode time with
        (b) the subdir-scoped meta returned by
        :meth:`~BaseSerializer.write` and recorded in the manifest's
        ``dirs`` section.  Subdir meta wins on key collisions because it
        represents what was actually written to disk.
        """
        dirs_entry = self._manifest.get("dirs", {}).get(node.subdir, {})
        merged_meta = {**node.meta, **dirs_entry.get("meta", {})}
        return owner.read(self._root / "dirs" / node.subdir, meta=merged_meta)


def _node_id(node: Node) -> str | None:
    if isinstance(node, Ref):
        return node.ref_id
    return node.node_id


def _with_node_id(node: Node, node_id: str) -> Node:
    if isinstance(node, Ref):
        msg = "Serializers must return concrete nodes, not Ref nodes."
        raise SerializationError(msg)
    return replace(node, node_id=node_id)
