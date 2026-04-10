"""Serializer for LightGBM models via native text format."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    qualified_type_name,
    write_meta,
)

__all__ = ["lightgbm_serializers", "lightgbm_serializers_by_type"]

lightgbm_serializers: SerializerTypeList = []
lightgbm_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("lightgbm") is not None:
    import importlib
    from pathlib import Path

    class LightGBMModelSerializer(Serializer[Any]):
        """Serialize LightGBM models via the native text format."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import lightgbm as lgb

            return isinstance(obj, (lgb.Booster, lgb.LGBMModel))

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import lightgbm as lgb

            model_path = str(directory / "model.txt")

            if isinstance(obj, lgb.LGBMModel):
                obj.booster_.save_model(model_path)
            else:
                obj.save_model(model_path)

            write_meta(
                directory,
                LightGBMModelSerializer,
                lightgbm_version=lgb.__version__,
                model_type=qualified_type_name(type(obj)),
            )

        @staticmethod
        def load(directory: Path) -> Any:
            import lightgbm as lgb

            from misen.utils.serde.serializer_base import read_meta

            meta = read_meta(directory)
            if meta is None:
                msg = "LightGBMModelSerializer requires serde_meta.json"
                raise ValueError(msg)

            model_path = str(directory / "model.txt")
            model_type = meta["model_type"]

            # Booster loads directly.
            if model_type == qualified_type_name(lgb.Booster):
                return lgb.Booster(model_file=model_path)

            # Sklearn-API wrappers need instantiation then loading the booster.
            module_name, _, attr_name = model_type.rpartition(".")
            module = importlib.import_module(module_name)
            cls = getattr(module, attr_name)
            model = cls()
            model._Booster = lgb.Booster(model_file=model_path)
            return model

    lightgbm_serializers = [LightGBMModelSerializer]
    lightgbm_serializers_by_type = {
        "lightgbm.basic.Booster": LightGBMModelSerializer,
    }
