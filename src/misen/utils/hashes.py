from misen_hash import canonical_hash
from typing_extensions import Self

__all__ = ["Hash", "ResolvedTaskHash", "ResultHash", "TaskHash", "short_hash"]


class Hash(int):
    """Integer hash with serialization helpers."""

    @classmethod
    def from_object(cls, obj: object) -> Self:
        """Create a hash from an arbitrary object."""
        return cls(canonical_hash(obj))

    def encode(self) -> bytes:
        """Encode the hash as big-endian bytes."""
        return self.to_bytes(8, "big", signed=False)

    @classmethod
    def decode(cls, b: bytes) -> Self:
        """Decode a hash from big-endian bytes."""
        return cls.from_bytes(b, "big", signed=False)

    def hex(self) -> str:
        """Return the hash as a fixed-width hexadecimal string."""
        return format(self, "016x")


class TaskHash(Hash):
    """Hash identifying a task by dependency structure."""


class ResolvedTaskHash(Hash):
    """Hash identifying a task with resolved inputs."""


class ResultHash(Hash):
    """Hash identifying a task result payload."""


def short_hash(obj: object) -> str:
    """Return a short hexadecimal hash for display."""
    return f"{hash(obj) & 0xFFFF:04x}"
