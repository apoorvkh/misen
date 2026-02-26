"""Public canonical hashing API and deterministic encoding helper."""

import io
from collections.abc import Callable
from typing import Any

import msgspec
from xxhash import xxh3_64, xxh3_64_intdigest

__all__ = ["hash_msgspec", "incremental_hash"]


def hash_msgspec(obj: Any) -> int:
    """Encode with deterministic msgspec JSON and hash with xxh3-64."""
    return xxh3_64_intdigest(msgspec.json.encode(obj, order="deterministic"), seed=0)


def incremental_hash(serialize: Callable[[io.RawIOBase], Any], seed: int = 0) -> int:
    """Hash serialized bytes incrementally as a serializer writes to a sink."""
    sink = _IncrementalHashWriter(seed=seed)
    serialize(sink)
    return sink.intdigest()


class _IncrementalHashWriter(io.RawIOBase):
    """Writable sink that updates an xxh3 hasher from streamed bytes."""

    def __init__(self, *, seed: int = 0) -> None:
        self._hasher = xxh3_64(seed=seed)

    def writable(self) -> bool:
        return True

    def write(self, b: Any, /) -> int:
        chunk = memoryview(b)
        self._hasher.update(chunk)
        return chunk.nbytes

    def intdigest(self) -> int:
        return self._hasher.intdigest()
