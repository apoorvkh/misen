"""Top-level ``save`` / ``load`` and the serializer :class:`Registry`.

:func:`save` performs a two-phase write:

1. **Encode walk** — recurses through the object with an
   :class:`EncodeCtx`, building a :class:`Node` graph and collecting
   leaf payloads grouped by ``leaf_kind``.
2. **Commit** — for each kind, the owning serializer's
   :meth:`~Serializer.write_batch` writes *all* payloads of that kind
   into one blob.  Directory-owning serializers each get their own
   subdirectory.

The whole graph and the per-kind metadata are recorded in a single
``manifest.json`` — no per-node meta files.
"""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from misen.exceptions import SerializationError
from misen.utils.serde.base import (
    BaseSerializer,
    Container,
    DecodeCtx,
    DirectoryLeaf,
    EncodeCtx,
    Leaf,
    Node,
    Ref,
)
from misen.utils.type_registry import TypeDispatchRegistry, qualified_type_name

__all__ = ["MANIFEST_FILENAME", "Registry", "load", "save"]

MANIFEST_FILENAME = "manifest.json"
_MANIFEST_VERSION = 2
_SUPPORTED_MANIFEST_VERSIONS = {1, _MANIFEST_VERSION}


class Registry:
    """Ordered list of v2 serializers plus fast dispatch.

    Internally delegates :meth:`lookup` to
    :class:`TypeDispatchRegistry`, which provides:

    1. A type-keyed cache so repeated saves of the same concrete type
       are O(1).
    2. An MRO walk over ``by_type_name`` so ``nn.Linear`` dispatches to
       ``TorchModuleSerializer`` via its ``nn.Module`` base.
    3. A linear ``match`` scan for content-dependent types (anything
       listed in ``volatile_types`` skips cache + MRO and re-evaluates
       the predicate every call).

    Pass ``by_type_name=None`` / ``volatile_types=None`` to get pure
    linear-scan dispatch (useful in tests).
    """

    def __init__(
        self,
        serializers: list[type[BaseSerializer]],
        *,
        by_type_name: Mapping[str, type[BaseSerializer]] | None = None,
        volatile_types: Iterable[type] | None = None,
    ) -> None:
        self._by_name: dict[str, type[BaseSerializer]] = {qualified_type_name(s): s for s in serializers}
        self._dispatch: TypeDispatchRegistry[type[BaseSerializer]] = TypeDispatchRegistry(
            by_type_name=by_type_name or {},
            candidates=serializers,
            predicate=lambda ser_cls, obj: ser_cls.match(obj),
            volatile_types=volatile_types,
        )

    def lookup(self, obj: Any) -> type[BaseSerializer] | None:
        """Return the serializer for *obj*, or ``None`` if none match."""
        return self._dispatch.lookup(obj)

    def by_name(self, name: str) -> type[BaseSerializer]:
        """Look up a serializer by qualified name (from a manifest)."""
        try:
            return self._by_name[name]
        except KeyError:
            msg = f"Unknown v2 serializer {name!r}.  It may have been renamed or removed."
            raise SerializationError(msg) from None


# ---------------------------------------------------------------------------
# Node ↔ JSON
# ---------------------------------------------------------------------------


def _node_to_json(node: Node) -> dict[str, Any]:
    if isinstance(node, Ref):
        return {"_t": "ref", "target": node.ref_id}
    if isinstance(node, Leaf):
        out: dict[str, Any] = {
            "_t": "leaf",
            "serializer": node.serializer,
            "kind": node.kind,
            "id": node.leaf_id,
            "meta": dict(node.meta),
        }
    elif isinstance(node, DirectoryLeaf):
        out = {
            "_t": "dir",
            "serializer": node.serializer,
            "subdir": node.subdir,
            "meta": dict(node.meta),
        }
    elif isinstance(node, Container):
        children = node.children
        if isinstance(children, Mapping):
            encoded_children: Any = {k: _node_to_json(v) for k, v in children.items()}
        else:
            encoded_children = [_node_to_json(v) for v in children]
        out = {
            "_t": "container",
            "serializer": node.serializer,
            "children": encoded_children,
            "meta": dict(node.meta),
        }
    else:
        msg = f"Cannot serialize node of type {type(node).__name__!r}"
        raise TypeError(msg)
    if node.node_id is not None:
        out["node_id"] = node.node_id
    return out


def _node_from_json(obj: dict[str, Any]) -> Node:
    kind_tag = obj["_t"]
    if kind_tag == "leaf":
        return Leaf(
            serializer=obj["serializer"],
            kind=obj["kind"],
            leaf_id=obj["id"],
            meta=obj.get("meta", {}),
            node_id=obj.get("node_id"),
        )
    if kind_tag == "dir":
        return DirectoryLeaf(
            serializer=obj["serializer"],
            subdir=obj["subdir"],
            meta=obj.get("meta", {}),
            node_id=obj.get("node_id"),
        )
    if kind_tag == "container":
        raw_children = obj["children"]
        if isinstance(raw_children, Mapping):
            children: Any = {k: _node_from_json(v) for k, v in raw_children.items()}
        else:
            children = [_node_from_json(v) for v in raw_children]
        return Container(
            serializer=obj["serializer"],
            children=children,
            meta=obj.get("meta", {}),
            node_id=obj.get("node_id"),
        )
    if kind_tag == "ref":
        return Ref(ref_id=obj["target"])
    msg = f"Unknown node tag {kind_tag!r} in manifest"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save(
    obj: Any,
    directory: Path,
    *,
    registry: Registry | None = None,
    ser_cls: type[BaseSerializer] | None = None,
) -> None:
    """Serialize *obj* into *directory*.

    Encodes the object into a :class:`Node` graph, writes a batched blob
    per leaf kind, and records everything in ``manifest.json``.

    Args:
        obj: Value to serialize.
        directory: Existing directory to write into.
        registry: Optional custom serializer registry.
        ser_cls: Optional explicit serializer class to use at the root
            — bypasses registry lookup for *obj*.  Children of *obj*
            still dispatch through the registry.  Matches the v1
            per-task ``@meta(serializer=...)`` override contract.
    """
    if registry is None:
        from misen.utils.serde.libs import default_registry

        registry = default_registry()

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    ctx = EncodeCtx(registry)
    root_node = ctx.encode(obj, ser_cls=ser_cls) if ser_cls is not None else ctx.encode(obj)

    # Commit batched leaves, one blob per kind.
    leaves_section: dict[str, dict[str, Any]] = {}
    if ctx.leaves_by_kind:
        leaves_root = directory / "leaves"
        leaves_root.mkdir(exist_ok=True)
        for kind, entries in ctx.leaves_by_kind.items():
            owner = ctx.leaf_owners[kind]
            kind_dir = leaves_root / kind
            kind_dir.mkdir(exist_ok=True)
            kind_meta = owner.write_batch(list(entries), kind_dir) or {}
            leaves_section[kind] = {
                "serializer": qualified_type_name(owner),
                "meta": dict(kind_meta),
            }

    # Commit directory-owning leaves, one subdir each.
    dirs_section: dict[str, dict[str, Any]] = {}
    if ctx.directory_leaves:
        dirs_root = directory / "dirs"
        dirs_root.mkdir(exist_ok=True)
        for owner, subdir, payload in ctx.directory_leaves:
            subdir_path = dirs_root / subdir
            subdir_path.mkdir(exist_ok=True)
            sub_meta = owner.write(payload, subdir_path) or {}
            dirs_section[subdir] = {
                "serializer": qualified_type_name(owner),
                "meta": dict(sub_meta),
            }

    manifest = {
        "version": _MANIFEST_VERSION,
        "root": _node_to_json(root_node),
        "leaves": leaves_section,
        "dirs": dirs_section,
    }
    (directory / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load(
    directory: Path,
    *,
    registry: Registry | None = None,
    ser_cls: type[BaseSerializer] | None = None,
) -> Any:
    """Deserialize whatever was stored in *directory* by :func:`save`.

    Args:
        directory: Directory written by a previous :func:`save` call.
        registry: Optional custom serializer registry.
        ser_cls: Optional explicit serializer class to use at the root
            — overrides the class named in the manifest.  Useful when
            the original class has been renamed but remains
            wire-compatible.  Children dispatch normally via the
            registry.
    """
    if registry is None:
        from misen.utils.serde.libs import default_registry

        registry = default_registry()

    directory = Path(directory)
    try:
        manifest: dict[str, Any] = json.loads((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except FileNotFoundError:
        msg = f"No {MANIFEST_FILENAME} found in {directory}"
        raise SerializationError(msg) from None

    version = manifest.get("version")
    if version not in _SUPPORTED_MANIFEST_VERSIONS:
        msg = (
            f"Unsupported {MANIFEST_FILENAME} version {version!r} in {directory} "
            f"(this build supports versions {sorted(_SUPPORTED_MANIFEST_VERSIONS)})."
        )
        raise SerializationError(msg)

    ctx = DecodeCtx(registry, directory, manifest)
    root_node = _node_from_json(manifest["root"])
    if ser_cls is not None:
        return ser_cls.decode(root_node, ctx)
    return ctx.decode(root_node)
