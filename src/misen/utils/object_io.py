import importlib.metadata
import importlib.util
import mmap
import pickletools
from abc import ABC, abstractmethod
from pathlib import Path
from pickle import UnpicklingError
from typing import Any, BinaryIO, Generic, TypeVar

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
        try:
            return dill.loads((dir / "data.dill").read_bytes())
        except UnpicklingError:
            raise ValueError(f"Failed to load object from {dir}")
            # TODO: compare environment against _dill_required_libs


# def dill_dump_with_required_modules(obj: Any, file: BinaryIO, *, protocol: int | None = None) -> set[str]:
#     """
#     Dump `obj` to `file` with dill and return a best-effort set of module names that
#     the unpickler will need to import/resolve (i.e., referenced globals).

#     This is usually faster than scanning the output afterward *if you're dumping anyway*.
#     """
#     required: set[str] = set()

#     class RecordingPickler(dill.Pickler):
#         # dill/pickle typically route "by reference" objects through save_global
#         def save_global(self, obj: Any, name: str | None = None, pack: Any = None) -> None:
#             mod = getattr(obj, "__module__", None)
#             if isinstance(mod, str) and mod:
#                 required.add(mod)
#             return super().save_global(obj, name=name, pack=pack)

#     p = RecordingPickler(file, protocol=protocol) if protocol is not None else RecordingPickler(file)
#     p.dump(obj)
#     return required
