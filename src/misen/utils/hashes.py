"""Hash utilities for tasks and results."""

import base64

from misen_hash import canonical_hash
from typing_extensions import Self

__all__ = ["Hash", "ResolvedTaskHash", "ResultHash", "TaskHash"]


class Hash(int):
    """Integer hash with serialization helpers."""

    @classmethod
    def from_object(cls, obj: object) -> Self:
        """Create an 8-byte hash from an arbitrary object."""
        # canonical_hash uses xxh3_64_intdigest -> returns int in [0, 2**64)
        return cls(canonical_hash(obj))

    def encode(self) -> bytes:
        """Encode the hash as big-endian bytes (8 bytes)."""
        v = int(self)
        if not (0 <= v < (1 << 64)):
            msg = f"Hash out of uint64 range: {v}"
            raise ValueError(msg)
        return v.to_bytes(8, "big", signed=False)

    @classmethod
    def decode(cls, b: bytes) -> Self:
        """Decode a hash from big-endian bytes (expects 8 bytes)."""
        if len(b) != 8:  # noqa: PLR2004
            msg = f"{cls}.decode expects 8 bytes, got {len(b)}"
            raise ValueError(msg)
        return cls(int.from_bytes(b, "big", signed=False))

    def b32(self) -> str:
        """Unpadded RFC 4648 base32 of this 64-bit hash. Always 13 chars."""
        return base64.b32encode(self.encode()).decode("ascii").rstrip("=")

    def short_b32(self) -> str:
        """First 4 chars of base32. Much higher collision probability."""
        return self.b32()[:4]


class TaskHash(Hash):
    """Hash identifying a task by dependency structure."""


class ResolvedTaskHash(Hash):
    """Hash identifying a task with resolved inputs."""


class ResultHash(Hash):
    """Hash identifying a task result payload."""
