"""Handlers for nltk tree and grammar objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry

__all__ = ["nltk_handlers", "nltk_handlers_by_type"]


nltk_handlers: HandlerTypeList = []
nltk_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("nltk") is not None:

    class NLTKTreeHandler(CollectionHandler):
        """Hash nltk.Tree objects by label and recursive children."""

        @staticmethod
        def match(obj: Any) -> bool:
            import nltk

            return isinstance(obj, nltk.Tree)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            return [obj.label(), list(obj)]

    class NLTKCFGHandler(CollectionHandler):
        """Hash nltk CFG objects by start symbol and productions."""

        @staticmethod
        def match(obj: Any) -> bool:
            import nltk.grammar

            return isinstance(obj, nltk.grammar.CFG)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            return [
                str(obj.start()),
                sorted(str(production) for production in obj.productions()),
            ]

    nltk_handlers = [NLTKTreeHandler, NLTKCFGHandler]
    nltk_handlers_by_type = {
        "nltk.tree.tree.Tree": NLTKTreeHandler,
        "nltk.grammar.CFG": NLTKCFGHandler,
    }
