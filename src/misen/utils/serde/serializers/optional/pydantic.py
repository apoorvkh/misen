"""Serializer for pydantic BaseModel instances."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    qualified_type_name,
    write_meta,
)

__all__ = ["pydantic_serializers", "pydantic_serializers_by_type"]

pydantic_serializers: SerializerTypeList = []
pydantic_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("pydantic") is not None:
    from pathlib import Path

    def _is_pydantic_model(obj: Any) -> bool:
        return any(
            base.__name__ == "BaseModel" and base.__module__.split(".")[0] == "pydantic"
            for base in type(obj).__mro__
        )

    class PydanticModelSerializer(Serializer[Any]):
        """Serialize pydantic ``BaseModel`` via ``model_dump_json``/``model_validate_json``."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            return _is_pydantic_model(obj)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            json_bytes = obj.model_dump_json().encode("utf-8")
            (directory / "data.json").write_bytes(json_bytes)
            write_meta(directory, PydanticModelSerializer, model_type=qualified_type_name(type(obj)))

        @staticmethod
        def load(directory: Path) -> Any:
            import importlib

            from misen.utils.serde.serializer_base import read_meta

            meta = read_meta(directory)
            if meta is None:
                msg = "PydanticModelSerializer requires serde_meta.json"
                raise ValueError(msg)

            module_name, _, attr_name = meta["model_type"].rpartition(".")
            module = importlib.import_module(module_name)
            cls = getattr(module, attr_name)

            json_data = (directory / "data.json").read_bytes()
            return cls.model_validate_json(json_data)

    pydantic_serializers = [PydanticModelSerializer]
    pydantic_serializers_by_type = {"pydantic.main.BaseModel": PydanticModelSerializer}
