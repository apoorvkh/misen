from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

import dill
import msgspec.msgpack

__all__ = ["Serializer", "DefaultSerializer"]

T = TypeVar("T")


class Serializer(ABC, Generic[T]):
    """Serialize/deserialize an object to/from a directory."""

    @staticmethod
    @abstractmethod
    def save(obj: T, dir: Path) -> None: ...

    @staticmethod
    @abstractmethod
    def load(dir: Path) -> T: ...


class DefaultSerializer(Serializer[Any]):
    @staticmethod
    def save(obj: Any, dir: Path) -> None:
        try:
            (dir / "data.msgpack").write_bytes(msgspec.msgpack.encode(obj))
        except NotImplementedError:
            (dir / "data.dill").write_bytes(dill.dumps(obj))

    @staticmethod
    def load(dir: Path) -> Any:
        if (dir / "data.msgpack").exists():
            return msgspec.msgpack.decode((dir / "data.msgpack").read_bytes())
        return dill.loads((dir / "data.dill").read_bytes())
