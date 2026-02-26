"""Handlers for Altair chart objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry

__all__ = ["altair_handlers", "altair_handlers_by_type"]


altair_handlers: HandlerTypeList = []
altair_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("altair") is not None:

    def _is_altair_chart(obj: Any) -> bool:
        if type(obj).__module__.split(".")[0] != "altair":
            return False

        return any(base.__name__ == "TopLevelMixin" for base in type(obj).__mro__)

    class AltairChartHandler(CollectionHandler):
        """Hash Altair chart objects by deterministic Vega-Lite dictionary."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _is_altair_chart(obj)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            payload = obj.to_dict() if hasattr(obj, "to_dict") else str(obj)
            return [payload]

    altair_handlers = [AltairChartHandler]
    altair_handlers_by_type = {"altair.vegalite.v5.api.TopLevelMixin": AltairChartHandler}
