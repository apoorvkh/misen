"""Recursive serializer for pydantic ``BaseModel`` instances.

Walks ``model_fields`` and dispatches each field value via
``ctx.encode``, so fields containing library-native types (tensors,
ndarrays, DataFrames, nested pydantic models, ...) round-trip through
their own serializers — v1's ``model_dump_json`` path raises on any
type pydantic's JSON encoder doesn't know.

Reconstruction on read uses direct instantiation (``cls(**fields)``)
which still runs pydantic's validators — so type mismatches caused by
a broken round-trip surface clearly rather than silently.
"""

import importlib.util
from typing import Any

from misen.utils.serde.base import BaseSerializer, Container, DecodeCtx, EncodeCtx, Node
from misen.utils.type_registry import import_by_qualified_name, qualified_type_name

__all__ = ["pydantic_serializers", "pydantic_serializers_by_type"]

pydantic_serializers: list[type[BaseSerializer]] = []
pydantic_serializers_by_type: dict[str, type[BaseSerializer]] = {}


if importlib.util.find_spec("pydantic") is not None:

    def _is_pydantic_model(obj: Any) -> bool:
        return any(
            base.__name__ == "BaseModel" and base.__module__.split(".")[0] == "pydantic" for base in type(obj).__mro__
        )

    class PydanticModelSerializer(BaseSerializer[Any]):
        """Recursive serializer for pydantic models — walks declared fields."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _is_pydantic_model(obj)

        @classmethod
        def encode(cls, obj: Any, ctx: EncodeCtx) -> Node:
            # ``model_fields`` lists declared fields only — computed
            # fields (``@computed_field``) are derived on access and
            # don't need round-tripping.
            children = {name: ctx.encode(getattr(obj, name)) for name in type(obj).model_fields}
            return Container(
                serializer=qualified_type_name(cls),
                children=children,
                meta={"model_type": qualified_type_name(type(obj))},
            )

        @classmethod
        def decode(cls, node: Node, ctx: DecodeCtx) -> Any:
            assert isinstance(node, Container)
            model_cls = import_by_qualified_name(node.meta["model_type"])
            field_values = {k: ctx.decode(v) for k, v in node.children.items()}
            # ``cls(**fields)`` re-runs validators, which is the stricter
            # of the available constructors (vs. ``model_construct``).
            # If a round-trip ever silently corrupts a field, validation
            # will catch it.
            return model_cls(**field_values)

    pydantic_serializers = [PydanticModelSerializer]
    pydantic_serializers_by_type = {"pydantic.main.BaseModel": PydanticModelSerializer}
