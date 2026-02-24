"""Handlers for statsmodels model/result objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry

__all__ = ["statsmodels_handlers", "statsmodels_handlers_by_type"]


statsmodels_handlers: HandlerTypeList = []
statsmodels_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("statsmodels") is not None:

    def _is_statsmodels_results(obj: Any) -> bool:
        if type(obj).__module__.split(".")[0] != "statsmodels":
            return False
        return hasattr(obj, "model") and hasattr(obj, "params")

    def _is_statsmodels_model(obj: Any) -> bool:
        if type(obj).__module__.split(".")[0] != "statsmodels":
            return False
        return hasattr(obj, "fit") and hasattr(obj, "endog") and hasattr(obj, "exog")

    class StatsmodelsResultsHandler(CollectionHandler):
        """Hash statsmodels fitted result wrappers by core summary vectors."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _is_statsmodels_results(obj)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            model = getattr(obj, "model", None)
            return [
                None if model is None else model.__class__.__module__,
                None if model is None else model.__class__.__qualname__,
                getattr(model, "endog_names", None),
                getattr(model, "exog_names", None),
                getattr(obj, "params", None),
                getattr(obj, "bse", None),
                getattr(obj, "pvalues", None),
                getattr(obj, "fittedvalues", None),
                getattr(obj, "resid", None),
                getattr(obj, "aic", None),
                getattr(obj, "bic", None),
                getattr(obj, "llf", None),
                getattr(obj, "nobs", None),
            ]

    class StatsmodelsModelHandler(CollectionHandler):
        """Hash statsmodels model specs by class and endog/exog payload."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _is_statsmodels_model(obj)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            return [
                getattr(obj, "endog", None),
                getattr(obj, "exog", None),
                getattr(obj, "formula", None),
                getattr(obj, "missing", None),
            ]

    statsmodels_handlers = [StatsmodelsResultsHandler, StatsmodelsModelHandler]
    statsmodels_handlers_by_type = {
        "statsmodels.regression.linear_model.RegressionResultsWrapper": StatsmodelsResultsHandler,
        "statsmodels.base.model.Results": StatsmodelsResultsHandler,
        "statsmodels.base.model.Model": StatsmodelsModelHandler,
    }
