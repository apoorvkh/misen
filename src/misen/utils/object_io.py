"""Serialization helpers for task results."""

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
