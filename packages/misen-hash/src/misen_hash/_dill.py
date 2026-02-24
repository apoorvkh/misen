"""Fallback handler that hashes objects via dill byte serialization."""

from typing import Any

import dill
from xxhash import xxh3_64_intdigest

from misen_hash import PrimitiveHandler

__all__ = ["DillHandler"]


class DillHandler(PrimitiveHandler):
    """Catch-all handler used when no specialized handler matches."""

    @staticmethod
    def match(obj: Any) -> bool:
        """Always match; this handler is used only as the final fallback."""
        _ = obj
        return True

    @staticmethod
    def digest(obj: Any, element_hash: None = None) -> int:
        """Hash dill-serialized object bytes."""
        _ = element_hash
        return xxh3_64_intdigest(dill.dumps(obj), seed=0)
