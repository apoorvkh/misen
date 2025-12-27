from misen_serialization import canonical_hash
from typing_extensions import Self

__all__ = ["Hash", "TaskHash", "ResolvedTaskHash", "ResultHash", "short_hash"]


class Hash(int):
    @classmethod
    def from_object(cls, obj: object) -> Self:
        return cls(canonical_hash(obj))

    def encode(self) -> bytes:
        return self.to_bytes(8, "big", signed=False)

    @classmethod
    def decode(cls, b: bytes) -> Self:
        return cls.from_bytes(b, "big", signed=False)


class TaskHash(Hash): ...


class ResolvedTaskHash(Hash): ...


class ResultHash(Hash): ...


def short_hash(obj: object) -> str:
    return f"{hash(obj) & 0xFFFF:04x}"
