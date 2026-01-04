from typing import Any

import msgspec
from xxhash import xxh3_64_intdigest

__all__ = ["hash_msgpack"]


def hash_msgpack(obj: Any) -> int:
    return xxh3_64_intdigest(msgspec.msgpack.encode(obj, order="deterministic"), seed=0)
