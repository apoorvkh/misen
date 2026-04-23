"""Recursive serializer for :mod:`dataclasses` instances.

Walks :func:`dataclasses.fields` and dispatches each field value via
``ctx.encode``, so fields holding library-native types (tensors,
ndarrays, DataFrames, ...) round-trip through their own serializers.
Before this serializer existed, dataclasses only round-tripped through
the tagged msgpack path in ``libs/stdlib.py``, which fails the moment a
field holds something msgpack can't encode.

Nested dataclass-within-dataclass also works: each nested instance
dispatches through :class:`DataclassSerializer` again, producing a
container tree whose leaves are whatever specialized serializer handles
the terminal values.
"""

import dataclasses
from typing import Any

from misen.utils.serde.base import BaseSerializer, Container, DecodeCtx, EncodeCtx, Node
from misen.utils.type_registry import import_by_qualified_name, qualified_type_name

__all__ = ["dataclass_serializers", "dataclass_serializers_by_type"]


class DataclassSerializer(BaseSerializer[Any]):
    """Recursive serializer for ``@dataclass`` instances."""

    @staticmethod
    def match(obj: Any) -> bool:
        return dataclasses.is_dataclass(obj) and not isinstance(obj, type)

    @classmethod
    def encode(cls, obj: Any, ctx: EncodeCtx) -> Node:
        children = {f.name: ctx.encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        return Container(
            serializer=qualified_type_name(cls),
            children=children,
            meta={"dataclass_type": qualified_type_name(type(obj))},
        )

    @classmethod
    def decode(cls, node: Node, ctx: DecodeCtx) -> Any:
        assert isinstance(node, Container)
        dataclass_cls = import_by_qualified_name(node.meta["dataclass_type"])
        field_values = {k: ctx.decode(v) for k, v in node.children.items()}
        # ``init=False`` fields aren't accepted by ``__init__`` — they're
        # typically set by ``__post_init__``.  Restore the saved value
        # directly (via ``object.__setattr__`` so frozen dataclasses work)
        # so any post-construction mutation round-trips too.
        init_field_names = {f.name for f in dataclasses.fields(dataclass_cls) if f.init}
        init_kwargs = {k: v for k, v in field_values.items() if k in init_field_names}
        post_init_values = {k: v for k, v in field_values.items() if k not in init_field_names}
        obj = dataclass_cls(**init_kwargs)
        for name, value in post_init_values.items():
            object.__setattr__(obj, name, value)
        return obj


dataclass_serializers: list[type[BaseSerializer]] = [DataclassSerializer]
# Every dataclass has a distinct concrete type with no shared base, so
# dispatch via the linear ``match`` scan (same as attrs).
dataclass_serializers_by_type: dict[str, type[BaseSerializer]] = {}
