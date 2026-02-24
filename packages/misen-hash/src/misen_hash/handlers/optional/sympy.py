"""Handlers for sympy expressions."""

import importlib.util
from typing import Any

from misen_hash.handler_base import HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import hash_msgspec

__all__ = ["sympy_handlers", "sympy_handlers_by_type"]


sympy_handlers: HandlerTypeList = []
sympy_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("sympy") is not None:

    class SymPyBasicHandler(PrimitiveHandler):
        """Hash sympy expressions by canonical structural representation."""

        @staticmethod
        def match(obj: Any) -> bool:
            if type(obj).__module__.split(".")[0] != "sympy":
                return False

            import sympy

            return isinstance(obj, sympy.Basic)

        @staticmethod
        def digest(obj: Any) -> int:
            import sympy

            return hash_msgspec(sympy.srepr(obj))

    sympy_handlers = [SymPyBasicHandler]
    sympy_handlers_by_type = {"sympy.core.basic.Basic": SymPyBasicHandler}
