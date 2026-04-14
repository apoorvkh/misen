"""Serializer for msgspec Struct instances."""

import importlib.util
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import Serializer, SerializerTypeRegistry
from misen.utils.type_registry import qualified_type_name

__all__ = ["msgspec_struct_serializers", "msgspec_struct_serializers_by_type"]

msgspec_struct_serializers: list[type[Serializer]] = []
msgspec_struct_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("msgspec") is not None:
    import importlib
    from pathlib import Path

    import msgspec
    import msgspec.msgpack

    class MsgspecStructSerializer(Serializer[Any]):
        """Serialize ``msgspec.Struct`` via msgspec's own msgpack codec."""

        @staticmethod
        def match(obj: Any) -> bool:
            return isinstance(obj, msgspec.Struct)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            data = msgspec.msgpack.encode(obj)
            (directory / "data.msgpack").write_bytes(data)
            return {"struct_type": qualified_type_name(type(obj))}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            module_name, _, attr_name = meta["struct_type"].rpartition(".")
            module = importlib.import_module(module_name)
            cls = getattr(module, attr_name)

            data = (directory / "data.msgpack").read_bytes()
            return msgspec.msgpack.decode(data, type=cls)

    msgspec_struct_serializers = [MsgspecStructSerializer]
    msgspec_struct_serializers_by_type = {"msgspec.Struct": MsgspecStructSerializer}
