from abc import ABC, abstractmethod
from pathlib import Path
from pickle import UnpicklingError
from typing import Any, Generic, TypeVar

import dill

__all__ = ["DefaultSerializer", "Serializer"]

T = TypeVar("T")

# TODO: record object dependencies ; if error, check mismatch


class Serializer(ABC, Generic[T]):
    """Serialize/deserialize an object to/from a directory."""

    __slots__ = ()

    @staticmethod
    @abstractmethod
    def save(obj: T, directory: Path) -> None:
        """Serialize an object into the given directory."""
        ...

    @staticmethod
    @abstractmethod
    def load(directory: Path) -> T:
        """Deserialize an object from the given directory."""
        ...


# TODO: add cases for other formats


class DefaultSerializer(Serializer[Any]):
    """Serialize objects using dill."""

    __slots__ = ()

    @staticmethod
    def save(obj: Any, directory: Path) -> None:
        """Serialize an object to a dill file in the directory."""
        (directory / "data.dill").write_bytes(dill.dumps(obj))

    @staticmethod
    def load(directory: Path) -> Any:
        """Deserialize an object from a dill file in the directory."""
        try:
            return dill.loads((directory / "data.dill").read_bytes())  # noqa: S301
        except UnpicklingError:
            msg = f"Failed to load object from {directory}"
            raise ValueError(msg) from None
            # TODO: compare environment against _dill_required_libs


# import importlib.util
# import mmap
# import pickletools
# from typing import BinaryIO

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
