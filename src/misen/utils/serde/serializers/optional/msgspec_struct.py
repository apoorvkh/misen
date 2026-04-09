"""Serializer for msgspec Struct instances."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    qualified_type_name,
    write_meta,
)

__all__ = ["msgspec_struct_serializers", "msgspec_struct_serializers_by_type"]

msgspec_struct_serializers: SerializerTypeList = []
msgspec_struct_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("msgspec") is not None:
    import importlib
    from pathlib import Path

    import msgspec
    import msgspec.msgpack

    class MsgspecStructSerializer(Serializer[Any]):
        """Serialize ``msgspec.Struct`` via msgspec's own msgpack codec."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            return isinstance(obj, msgspec.Struct)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            data = msgspec.msgpack.encode(obj)
            (directory / "data.msgpack").write_bytes(data)
            write_meta(directory, MsgspecStructSerializer, struct_type=qualified_type_name(type(obj)))

        @staticmethod
        def load(directory: Path) -> Any:
            from misen.utils.serde.serializer_base import read_meta

            meta = read_meta(directory)
            if meta is None:
                msg = "MsgspecStructSerializer requires serde_meta.json"
                raise ValueError(msg)

            module_name, _, attr_name = meta["struct_type"].rpartition(".")
            module = importlib.import_module(module_name)
            cls = getattr(module, attr_name)

            data = (directory / "data.msgpack").read_bytes()
            return msgspec.msgpack.decode(data, type=cls)

    msgspec_struct_serializers = [MsgspecStructSerializer]
    msgspec_struct_serializers_by_type = {"msgspec.Struct": MsgspecStructSerializer}
