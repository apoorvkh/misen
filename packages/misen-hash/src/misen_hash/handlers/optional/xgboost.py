"""Handlers for xgboost booster and sklearn-style estimator objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import hash_msgspec, incremental_hash

__all__ = ["xgboost_handlers", "xgboost_handlers_by_type"]


xgboost_handlers: HandlerTypeList = []
xgboost_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("xgboost") is not None:

    def _has_xgboost_model_base(obj: Any) -> bool:
        return any(
            base.__name__ == "XGBModel" and base.__module__ == "xgboost.sklearn"
            for base in type(obj).__mro__
        )

    class XGBoostBoosterHandler(PrimitiveHandler):
        """Hash xgboost Booster payload plus schema metadata."""

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "xgboost":
                return False

            import xgboost

            return isinstance(obj, xgboost.Booster)

        @staticmethod
        def digest(obj: Any) -> int:
            payload = bytes(obj.save_raw())
            payload_hash = incremental_hash(lambda sink: sink.write(payload))
            return hash_msgspec(
                (
                    tuple(obj.feature_names) if obj.feature_names is not None else None,
                    tuple(obj.feature_types) if obj.feature_types is not None else None,
                    obj.save_config() if hasattr(obj, "save_config") else None,
                    payload_hash,
                )
            )

    class XGBoostSklearnModelHandler(CollectionHandler):
        """Hash xgboost sklearn models by params and optional fitted booster state."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_xgboost_model_base(obj)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            params = obj.get_params(deep=True)
            booster_payload_hash = None

            booster = getattr(obj, "_Booster", None)
            if booster is not None:
                booster_payload = bytes(booster.save_raw())
                booster_payload_hash = incremental_hash(lambda sink: sink.write(booster_payload))

            return [
                params,
                booster_payload_hash,
            ]

    xgboost_handlers = [
        XGBoostBoosterHandler,
        XGBoostSklearnModelHandler,
    ]
    xgboost_handlers_by_type = {
        "xgboost.core.Booster": XGBoostBoosterHandler,
        "xgboost.sklearn.XGBModel": XGBoostSklearnModelHandler,
    }
