"""Serializer for attrs classes with msgpack-safe fields."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    qualified_type_name,
    write_meta,
)

__all__ = ["attrs_serializers", "attrs_serializers_by_type"]

attrs_serializers: SerializerTypeList = []
attrs_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("attrs") is not None:
    import importlib
    from pathlib import Path

    import msgspec.msgpack

    from misen.utils.serde.serializers.stdlib import _decode_tagged, _encode_tagged, _is_msgpack_safe

    class AttrsSerializer(Serializer[Any]):
        """Serialize ``@attrs.define`` instances with msgpack-safe fields."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            if not hasattr(type(obj), "__attrs_attrs__"):
                return False
            import attrs

            try:
                return _is_msgpack_safe(attrs.asdict(obj))
            except (TypeError, ValueError):
                return False

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import attrs

            data = attrs.asdict(obj)
            tagged = _encode_tagged(data)
            encoded = msgspec.msgpack.encode(tagged)
            (directory / "data.msgpack").write_bytes(encoded)
            write_meta(directory, AttrsSerializer, attrs_type=qualified_type_name(type(obj)))

        @staticmethod
        def load(directory: Path) -> Any:
            from misen.utils.serde.serializer_base import read_meta

            meta = read_meta(directory)
            if meta is None:
                msg = "AttrsSerializer requires serde_meta.json"
                raise ValueError(msg)

            raw = msgspec.msgpack.decode((directory / "data.msgpack").read_bytes())
            data = _decode_tagged(raw)

            module_name, _, attr_name = meta["attrs_type"].rpartition(".")
            module = importlib.import_module(module_name)
            cls = getattr(module, attr_name)
            return cls(**data)

    attrs_serializers = [AttrsSerializer]
    attrs_serializers_by_type = {}  # attrs types vary, rely on match()
