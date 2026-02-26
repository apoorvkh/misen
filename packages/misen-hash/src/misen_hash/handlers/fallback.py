"""Fallback handler that hashes objects via dill byte serialization."""

from typing import Any

import dill

from misen_hash.handler_base import PrimitiveHandler
from misen_hash.hash import incremental_hash

__all__ = ["DillHandler"]


class DillHandler(PrimitiveHandler):
    """Catch-all handler used when no specialized handler matches."""

    @staticmethod
    def match(obj: Any) -> bool:
        """Always match; this handler is used only as the final fallback."""
        _ = obj
        return True

    @staticmethod
    def digest(obj: Any) -> int:
        """Hash dill-serialized object bytes."""
        return incremental_hash(lambda sink: dill.dump(obj, sink))
