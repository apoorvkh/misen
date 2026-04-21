"""Recursive serializer for ``msgspec.Struct`` instances.

Walks ``__struct_fields__`` and dispatches each field value via
``ctx.encode``, so struct fields holding types outside msgspec's
native set (tensors, ndarrays, nested Structs with such fields)
round-trip through their own serializers — v1's
``msgspec.msgpack.encode`` path raises on anything msgspec itself
can't encode.

Limitation: tagged unions and custom ``encode_hook``/``dec_hook`` are
not exercised through this path.  Structs that rely on those should
keep using v1 semantics (or the user can register a more specific
serializer with higher priority).
"""

import importlib
import importlib.util
from typing import Any

from misen.utils.serde.base import BaseSerializer, Container, DecodeCtx, EncodeCtx, Node
from misen.utils.type_registry import qualified_type_name

__all__ = ["msgspec_struct_serializers", "msgspec_struct_serializers_by_type"]

msgspec_struct_serializers: list[type[BaseSerializer]] = []
msgspec_struct_serializers_by_type: dict[str, type[BaseSerializer]] = {}


if importlib.util.find_spec("msgspec") is not None:
    import msgspec

    class MsgspecStructSerializer(BaseSerializer[Any]):
        """Recursive serializer for msgspec Structs — walks declared fields."""

        @staticmethod
        def match(obj: Any) -> bool:
            return isinstance(obj, msgspec.Struct)

        @classmethod
        def encode(cls, obj: Any, ctx: EncodeCtx) -> Node:
            children = {name: ctx.encode(getattr(obj, name)) for name in type(obj).__struct_fields__}
            return Container(
                serializer=qualified_type_name(cls),
                children=children,
                meta={"struct_type": qualified_type_name(type(obj))},
            )

        @classmethod
        def decode(cls, node: Node, ctx: DecodeCtx) -> Any:
            assert isinstance(node, Container)
            module_name, _, attr_name = node.meta["struct_type"].rpartition(".")
            module = importlib.import_module(module_name)
            struct_cls = getattr(module, attr_name)
            field_values = {k: ctx.decode(v) for k, v in node.children.items()}
            return struct_cls(**field_values)

    msgspec_struct_serializers = [MsgspecStructSerializer]
    msgspec_struct_serializers_by_type = {"msgspec.Struct": MsgspecStructSerializer}
