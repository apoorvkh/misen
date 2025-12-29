from typing import Any

import dill
import msgspec.msgpack

__all__ = ["to_bytes", "from_bytes"]


def to_bytes(obj: Any) -> bytes:
    try:
        return b"M" + msgspec.msgpack.encode(obj)
    except NotImplementedError:
        return b"D" + dill.dumps(obj)


def from_bytes(data: bytes) -> Any:
    match data[:1]:
        case b"M":
            return msgspec.msgpack.decode(data[1:])
        case b"D":
            return dill.loads(data[1:])
    raise ValueError("Invalid data")
