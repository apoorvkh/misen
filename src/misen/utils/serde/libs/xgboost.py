"""Serializer for XGBoost models via native save/load."""

import importlib.util
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import Serializer, SerializerTypeRegistry
from misen.utils.type_registry import qualified_type_name

__all__ = ["xgboost_serializers", "xgboost_serializers_by_type"]

xgboost_serializers: list[type[Serializer]] = []
xgboost_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("xgboost") is not None:
    import importlib
    from pathlib import Path

    class XGBoostModelSerializer(Serializer[Any]):
        """Serialize XGBoost models via the native UBJSON/JSON format."""

        @staticmethod
        def match(obj: Any) -> bool:
            import xgboost as xgb

            return isinstance(obj, (xgb.Booster, xgb.XGBModel))

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import xgboost as xgb

            obj.save_model(str(directory / "model.ubj"))
            return {
                "xgboost_version": xgb.__version__,
                "model_type": qualified_type_name(type(obj)),
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            import xgboost as xgb

            model_type = meta["model_type"]
            model_path = str(directory / "model.ubj")

            # Booster loads directly.
            if model_type == qualified_type_name(xgb.Booster):
                return xgb.Booster(model_file=model_path)

            # Sklearn-API wrappers need instantiation then load.
            module_name, _, attr_name = model_type.rpartition(".")
            module = importlib.import_module(module_name)
            cls = getattr(module, attr_name)
            model = cls()
            model.load_model(model_path)
            return model

    xgboost_serializers = [XGBoostModelSerializer]
    xgboost_serializers_by_type = {
        "xgboost.core.Booster": XGBoostModelSerializer,
        "xgboost.sklearn.XGBModel": XGBoostModelSerializer,
    }
