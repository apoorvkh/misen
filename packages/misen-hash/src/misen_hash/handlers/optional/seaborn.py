"""Handlers for seaborn grid objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry

__all__ = ["seaborn_handlers", "seaborn_handlers_by_type"]


seaborn_handlers: HandlerTypeList = []
seaborn_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("seaborn") is not None:

    def _is_seaborn_grid(obj: Any) -> bool:
        if type(obj).__module__.split(".")[0] != "seaborn":
            return False

        return any(
            base.__name__ in {"FacetGrid", "PairGrid", "JointGrid", "ClusterGrid"}
            for base in type(obj).__mro__
        )

    class SeabornGridHandler(CollectionHandler):
        """Hash seaborn grid objects by class, data payload, and underlying figure."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _is_seaborn_grid(obj)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            figure = getattr(obj, "figure", None)
            if figure is None:
                figure = getattr(obj, "fig", None)

            return [
                getattr(obj, "data", None),
                getattr(obj, "variables", None),
                getattr(obj, "row_names", None),
                getattr(obj, "col_names", None),
                getattr(obj, "hue_names", None),
                figure,
            ]

    seaborn_handlers = [SeabornGridHandler]
    seaborn_handlers_by_type = {
        "seaborn.axisgrid.FacetGrid": SeabornGridHandler,
        "seaborn.axisgrid.PairGrid": SeabornGridHandler,
        "seaborn.axisgrid.JointGrid": SeabornGridHandler,
        "seaborn.matrix.ClusterGrid": SeabornGridHandler,
    }
