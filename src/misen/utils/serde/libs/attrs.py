"""Serializer for attrs classes."""

import importlib.util
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import Serializer, SerializerTypeRegistry, UnserializableTypeError
from misen.utils.type_registry import qualified_type_name

__all__ = ["attrs_serializers", "attrs_serializers_by_type"]

attrs_serializers: list[type[Serializer]] = []
attrs_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("attrs") is not None:
    import importlib
    from pathlib import Path

    import msgspec.msgpack

    from misen.utils.serde.libs.stdlib import _decode_tagged, _encode_tagged

    class AttrsSerializer(Serializer[Any]):
        """Serialize ``@attrs.define`` instances by walking their fields.

        Fields are encoded recursively via :func:`_encode_tagged` so that
        nested dataclasses, attrs instances, and stdlib value types keep
        their type identity on round-trip.  If any field contains a value
        :mod:`msgspec.msgpack` cannot encode, ``write`` raises
        :class:`UnserializableTypeError` rather than crashing inside
        ``msgspec``.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            return hasattr(type(obj), "__attrs_attrs__") and not isinstance(obj, type)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            field_values = {a.name: _encode_tagged(getattr(obj, a.name)) for a in type(obj).__attrs_attrs__}
            try:
                encoded = msgspec.msgpack.encode(field_values)
            except (TypeError, OverflowError) as exc:
                msg = (
                    f"Cannot serialize attrs instance of type {qualified_type_name(type(obj))!r} "
                    f"with AttrsSerializer: {exc}"
                )
                raise UnserializableTypeError(msg) from exc
            (directory / "data.msgpack").write_bytes(encoded)
            return {"attrs_type": qualified_type_name(type(obj))}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            raw = msgspec.msgpack.decode((directory / "data.msgpack").read_bytes())
            field_values = {k: _decode_tagged(v) for k, v in raw.items()}

            module_name, _, attr_name = meta["attrs_type"].rpartition(".")
            module = importlib.import_module(module_name)
            cls = getattr(module, attr_name)
            return cls(**field_values)

    attrs_serializers = [AttrsSerializer]
    attrs_serializers_by_type = {}  # attrs types vary, rely on match()
