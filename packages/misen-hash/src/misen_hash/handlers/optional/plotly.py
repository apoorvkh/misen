"""Handlers for plotly figures."""

import importlib.util
from typing import Any

from misen_hash.handler_base import HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import hash_msgspec

__all__ = ["plotly_handlers", "plotly_handlers_by_type"]


plotly_handlers: HandlerTypeList = []
plotly_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("plotly") is not None:

    class PlotlyFigureHandler(PrimitiveHandler):
        """Hash plotly figures by JSON payload."""

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "plotly":
                return False

            return any(base.__name__ == "BaseFigure" for base in type(obj).__mro__)

        @staticmethod
        def digest(obj: Any) -> int:
            if hasattr(obj, "to_plotly_json"):
                return hash_msgspec(obj.to_plotly_json())
            return hash_msgspec(str(obj))

    plotly_handlers = [PlotlyFigureHandler]
    plotly_handlers_by_type = {"plotly.basedatatypes.BaseFigure": PlotlyFigureHandler}
