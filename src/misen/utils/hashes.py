"""Typed 64-bit hash wrappers for task/result identity."""

import base64

from misen_hash import canonical_hash
from typing_extensions import Self

__all__ = ["Hash", "ResolvedTaskHash", "ResultHash", "TaskHash"]


class Hash(int):
    """Unsigned 64-bit integer hash with encoding helpers."""

    @classmethod
    def from_object(cls, obj: object) -> Self:
        """Create typed hash from arbitrary object.

        Args:
            obj: Object to hash canonically.

        Returns:
            ``Hash`` subclass wrapping a 64-bit digest.
        """
        # canonical_hash returns unsigned xxh3-64 int digests.
        return cls(canonical_hash(obj))

    def encode(self) -> bytes:
        """Encode hash as big-endian 8-byte representation.

        Returns:
            Encoded bytes.

        Raises:
            ValueError: If integer value is outside uint64 range.
        """
        v = int(self)
        if not (0 <= v < (1 << 64)):
            msg = f"Hash out of uint64 range: {v}"
            raise ValueError(msg)
        return v.to_bytes(8, "big", signed=False)

    @classmethod
    def decode(cls, b: bytes) -> Self:
        """Decode hash from 8-byte big-endian representation.

        Args:
            b: Encoded bytes.

        Returns:
            Decoded hash object.

        Raises:
            ValueError: If input length is not 8 bytes.
        """
        if len(b) != 8:  # noqa: PLR2004
            msg = f"{cls}.decode expects 8 bytes, got {len(b)}"
            raise ValueError(msg)
        return cls(int.from_bytes(b, "big", signed=False))

    def b32(self) -> str:
        """Return unpadded RFC 4648 base32 string (13 chars)."""
        return base64.b32encode(self.encode()).decode("ascii").rstrip("=")

    def short_b32(self) -> str:
        """Return short base32 prefix for human-readable debug output."""
        return self.b32()[:4]


class TaskHash(Hash):
    """Hash identifying task structure and unresolved dependencies."""


class ResolvedTaskHash(Hash):
    """Hash identifying task inputs after dependency resolution."""


class ResultHash(Hash):
    """Hash identifying serialized result payload identity."""
