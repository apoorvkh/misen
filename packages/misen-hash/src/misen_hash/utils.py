from typing import Any

import msgspec
from xxhash import xxh3_64_intdigest

__all__ = ["hash_msgspec"]


def hash_msgspec(obj: Any) -> int:
    return xxh3_64_intdigest(msgspec.json.encode(obj, order="deterministic"), seed=0)
