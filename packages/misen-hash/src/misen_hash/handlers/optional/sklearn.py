"""Handlers for scikit-learn estimators."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry

__all__ = ["sklearn_handlers", "sklearn_handlers_by_type"]


sklearn_handlers: HandlerTypeList = []
sklearn_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("sklearn") is not None:

    def _is_sklearn_estimator(obj: Any) -> bool:
        return any(
            base.__name__ == "BaseEstimator" and base.__module__ == "sklearn.base"
            for base in type(obj).__mro__
        )

    class SklearnEstimatorHandler(CollectionHandler):
        """Hash sklearn estimator parameters and fitted trailing-underscore state."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _is_sklearn_estimator(obj)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            params = obj.get_params(deep=True)
            state = {key: value for key, value in vars(obj).items() if key.endswith("_") and not key.startswith("__")}
            return [obj.__class__.__module__, obj.__class__.__qualname__, params, state]

    sklearn_handlers = [SklearnEstimatorHandler]
    sklearn_handlers_by_type = {"sklearn.base.BaseEstimator": SklearnEstimatorHandler}
