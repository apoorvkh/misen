"""Serializer for CatBoost models via native CBM format."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    qualified_type_name,
    write_meta,
)

__all__ = ["catboost_serializers", "catboost_serializers_by_type"]

catboost_serializers: SerializerTypeList = []
catboost_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("catboost") is not None:
    import importlib
    from pathlib import Path

    class CatBoostModelSerializer(Serializer[Any]):
        """Serialize CatBoost models via the native CBM binary format."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import catboost

            return isinstance(obj, catboost.CatBoost)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import catboost

            obj.save_model(str(directory / "model.cbm"), format="cbm")
            write_meta(
                directory,
                CatBoostModelSerializer,
                catboost_version=catboost.__version__,
                model_type=qualified_type_name(type(obj)),
            )

        @staticmethod
        def load(directory: Path) -> Any:
            from misen.utils.serde.serializer_base import read_meta

            meta = read_meta(directory)
            if meta is None:
                msg = "CatBoostModelSerializer requires serde_meta.json"
                raise ValueError(msg)

            model_type = meta["model_type"]
            module_name, _, attr_name = model_type.rpartition(".")
            module = importlib.import_module(module_name)
            cls = getattr(module, attr_name)

            model = cls()
            model.load_model(str(directory / "model.cbm"), format="cbm")
            return model

    catboost_serializers = [CatBoostModelSerializer]
    catboost_serializers_by_type = {}  # CatBoost subclasses vary, rely on match()
