"""Serializer for CatBoost models via native CBM format."""

import importlib.util
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import Serializer, SerializerTypeRegistry
from misen.utils.type_registry import qualified_type_name

__all__ = ["catboost_serializers", "catboost_serializers_by_type"]

catboost_serializers: list[type[Serializer]] = []
catboost_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("catboost") is not None:
    import importlib
    from pathlib import Path

    class CatBoostModelSerializer(Serializer[Any]):
        """Serialize CatBoost models via the native CBM binary format."""

        @staticmethod
        def match(obj: Any) -> bool:
            import catboost

            return isinstance(obj, catboost.CatBoost)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import catboost

            obj.save_model(str(directory / "model.cbm"), format="cbm")
            return {
                "catboost_version": catboost.__version__,
                "model_type": qualified_type_name(type(obj)),
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            model_type = meta["model_type"]
            module_name, _, attr_name = model_type.rpartition(".")
            module = importlib.import_module(module_name)
            cls = getattr(module, attr_name)

            model = cls()
            model.load_model(str(directory / "model.cbm"), format="cbm")
            return model

    catboost_serializers = [CatBoostModelSerializer]
    catboost_serializers_by_type = {}  # CatBoost subclasses vary, rely on match()
