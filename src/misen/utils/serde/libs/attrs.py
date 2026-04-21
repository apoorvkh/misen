"""Recursive serializer for ``@attrs.define`` instances.

Walks ``__attrs_attrs__`` and dispatches each field value via
``ctx.encode``, so fields holding library-native types (tensors,
ndarrays, pandas DataFrames, ...) round-trip through their own
serializers — a capability the v1 flat serializer could not offer
because it funneled all fields through ``msgspec.msgpack``, which
knows nothing about e.g. tensors.

Nested attrs-within-attrs also works: each nested instance dispatches
through :class:`AttrsSerializer` again, producing a container tree
whose leaves are whatever specialized serializer handles the terminal
values.
"""

import importlib
import importlib.util
from typing import Any

from misen.utils.serde.base import BaseSerializer, Container, DecodeCtx, EncodeCtx, Node
from misen.utils.type_registry import qualified_type_name

__all__ = ["attrs_serializers", "attrs_serializers_by_type"]

attrs_serializers: list[type[BaseSerializer]] = []
attrs_serializers_by_type: dict[str, type[BaseSerializer]] = {}


if importlib.util.find_spec("attrs") is not None:

    class AttrsSerializer(BaseSerializer[Any]):
        """Recursive serializer for attrs-decorated classes."""

        @staticmethod
        def match(obj: Any) -> bool:
            return hasattr(type(obj), "__attrs_attrs__") and not isinstance(obj, type)

        @classmethod
        def encode(cls, obj: Any, ctx: EncodeCtx) -> Node:
            children = {a.name: ctx.encode(getattr(obj, a.name)) for a in type(obj).__attrs_attrs__}
            return Container(
                serializer=qualified_type_name(cls),
                children=children,
                meta={"attrs_type": qualified_type_name(type(obj))},
            )

        @classmethod
        def decode(cls, node: Node, ctx: DecodeCtx) -> Any:
            assert isinstance(node, Container)
            module_name, _, attr_name = node.meta["attrs_type"].rpartition(".")
            module = importlib.import_module(module_name)
            attrs_cls = getattr(module, attr_name)
            field_values = {k: ctx.decode(v) for k, v in node.children.items()}
            return attrs_cls(**field_values)

    attrs_serializers = [AttrsSerializer]
    # attrs classes have no shared base in ``by_type_name``; dispatch via
    # the linear ``match`` scan (there's no common ancestor, unlike
    # pydantic's BaseModel or msgspec.Struct).
    attrs_serializers_by_type = {}
