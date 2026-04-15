"""Serializer for pydantic BaseModel instances."""

import importlib.util
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import Serializer, SerializerTypeRegistry
from misen.utils.type_registry import qualified_type_name

__all__ = ["pydantic_serializers", "pydantic_serializers_by_type"]

pydantic_serializers: list[type[Serializer]] = []
pydantic_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("pydantic") is not None:
    from pathlib import Path

    def _is_pydantic_model(obj: Any) -> bool:
        return any(
            base.__name__ == "BaseModel" and base.__module__.split(".")[0] == "pydantic" for base in type(obj).__mro__
        )

    class PydanticModelSerializer(Serializer[Any]):
        """Serialize pydantic ``BaseModel`` via ``model_dump_json``/``model_validate_json``."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _is_pydantic_model(obj)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            json_bytes = obj.model_dump_json().encode("utf-8")
            (directory / "data.json").write_bytes(json_bytes)
            return {"model_type": qualified_type_name(type(obj))}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            import importlib

            module_name, _, attr_name = meta["model_type"].rpartition(".")
            module = importlib.import_module(module_name)
            cls = getattr(module, attr_name)

            json_data = (directory / "data.json").read_bytes()
            return cls.model_validate_json(json_data)

    pydantic_serializers = [PydanticModelSerializer]
    pydantic_serializers_by_type = {"pydantic.main.BaseModel": PydanticModelSerializer}
