"""Result serialization interfaces used by task caching."""

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
        """Serialize object into directory.

        Args:
            obj: Object to serialize.
            directory: Target directory.
        """

    @staticmethod
    @abstractmethod
    def load(directory: Path) -> T:
        """Deserialize object from directory.

        Args:
            directory: Source directory.

        Returns:
            Deserialized object.
        """


# TODO: add cases for other formats


class DefaultSerializer(Serializer[Any]):
    """Serializer that stores values as ``data.dill``."""

    __slots__ = ()

    @staticmethod
    def save(obj: Any, directory: Path) -> None:
        """Serialize object to ``data.dill`` in directory."""
        (directory / "data.dill").write_bytes(dill.dumps(obj))

    @staticmethod
    def load(directory: Path) -> Any:
        """Deserialize object from ``data.dill``.

        Args:
            directory: Directory containing serialized object.

        Returns:
            Deserialized object.

        Raises:
            ValueError: If dill payload cannot be unpickled.
        """
        try:
            return dill.loads((directory / "data.dill").read_bytes())  # noqa: S301
        except UnpicklingError:
            msg = f"Failed to load object from {directory}"
            raise ValueError(msg) from None
            # TODO: compare environment against _dill_required_libs
