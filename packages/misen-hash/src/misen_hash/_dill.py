from typing import Any

import dill
from xxhash import xxh3_64_intdigest

from . import PrimitiveHandler

__all__ = ["DillHandler"]


class DillHandler(PrimitiveHandler):
    @staticmethod
    def matches(obj: Any) -> bool:
        return True

    @staticmethod
    def digest(obj: Any, element_hash: None = None) -> int:
        return xxh3_64_intdigest(dill.dumps(obj), seed=0)
