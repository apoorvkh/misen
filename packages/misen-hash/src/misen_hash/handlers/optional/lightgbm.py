"""Handlers for lightgbm booster and sklearn-style estimator objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import hash_msgspec

__all__ = ["lightgbm_handlers", "lightgbm_handlers_by_type"]


lightgbm_handlers: HandlerTypeList = []
lightgbm_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("lightgbm") is not None:

    def _has_lightgbm_model_base(obj: Any) -> bool:
        return any(
            base.__name__ == "LGBMModel" and base.__module__ == "lightgbm.sklearn"
            for base in type(obj).__mro__
        )

    class LightGBMBoosterHandler(PrimitiveHandler):
        """Hash lightgbm Booster objects by serialized model text."""

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "lightgbm":
                return False

            import lightgbm as lgb

            return isinstance(obj, lgb.Booster)

        @staticmethod
        def digest(obj: Any) -> int:
            return hash_msgspec(obj.model_to_string(num_iteration=-1))

    class LightGBMSklearnModelHandler(CollectionHandler):
        """Hash lightgbm sklearn models by params and optional fitted booster state."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_lightgbm_model_base(obj)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            params = obj.get_params(deep=True)
            booster_payload = None

            booster = getattr(obj, "booster_", None)
            if booster is not None:
                booster_payload = booster.model_to_string(num_iteration=-1)

            return [
                params,
                booster_payload,
            ]

    lightgbm_handlers = [
        LightGBMBoosterHandler,
        LightGBMSklearnModelHandler,
    ]
    lightgbm_handlers_by_type = {
        "lightgbm.basic.Booster": LightGBMBoosterHandler,
        "lightgbm.sklearn.LGBMModel": LightGBMSklearnModelHandler,
    }
