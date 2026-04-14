"""Serializer for LightGBM models via native text format."""

import importlib.util
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import Serializer, SerializerTypeRegistry
from misen.utils.type_registry import qualified_type_name

__all__ = ["lightgbm_serializers", "lightgbm_serializers_by_type"]

lightgbm_serializers: list[type[Serializer]] = []
lightgbm_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("lightgbm") is not None:
    import importlib
    from pathlib import Path

    class LightGBMModelSerializer(Serializer[Any]):
        """Serialize LightGBM models via the native text format."""

        @staticmethod
        def match(obj: Any) -> bool:
            import lightgbm as lgb

            return isinstance(obj, (lgb.Booster, lgb.LGBMModel))

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import lightgbm as lgb

            model_path = str(directory / "model.txt")

            if isinstance(obj, lgb.LGBMModel):
                obj.booster_.save_model(model_path)
            else:
                obj.save_model(model_path)

            return {
                "lightgbm_version": lgb.__version__,
                "model_type": qualified_type_name(type(obj)),
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            import lightgbm as lgb

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
