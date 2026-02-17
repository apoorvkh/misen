"""Settings and singleton utilities for configurable components.

This module supports ``"auto"`` construction of executor/workspace instances
from ``misen.toml`` and memoizes struct instances by constructor kwargs for
lightweight singleton behavior.
"""

import os
import sys
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from functools import cache, cached_property
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar

import msgspec
from msgspec import Struct
from typing_extensions import Self

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


__all__ = ["FromSettingsABC", "Settings"]


DEFAULT_SETTINGS_FILE = Path(os.environ.get("MISEN_SETTINGS_FILE") or (Path.cwd() / "misen.toml"))


class Settings(Struct, dict=True):
    """Settings loader for TOML configuration files."""

    file: Path = DEFAULT_SETTINGS_FILE

    @cached_property
    def toml_data(self) -> dict[str, Any]:
        """Return parsed TOML settings data.

        Returns:
            Parsed TOML dictionary, or empty dict if file does not exist.
        """
        try:
            return tomllib.loads(self.file.read_bytes().decode())
        except FileNotFoundError:
            return {}

    def __hash__(self) -> int:
        """Return hash based on settings file identity and stat metadata."""
        settings_file = self.file.expanduser().resolve()
        try:
            stat = settings_file.stat()
            return hash((str(settings_file), stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            return hash(None)


class FromSettingsMeta(msgspec.StructMeta, ABCMeta):
    """Metaclass implementing a parameterized singleton cache."""

    _instances: ClassVar[dict[bytes, Any]] = {}

    def __call__(cls, **kwargs: Any) -> Any:
        """Return memoized instance for given constructor kwargs."""
        key = msgspec.json.encode((str(cls.__module__), str(cls.__qualname__), kwargs))
        if key not in FromSettingsMeta._instances:
            FromSettingsMeta._instances[key] = super().__call__(**kwargs)
        return FromSettingsMeta._instances[key]


class FromSettingsABC(msgspec.Struct, dict=True, metaclass=FromSettingsMeta):
    """Base class for settings-backed singleton structs."""

    @staticmethod
    @abstractmethod
    def _default() -> Self:
        """Return default instance when settings do not specify a type."""

    @staticmethod
    @abstractmethod
    def _settings_key() -> str:
        """Return TOML key prefix for this type."""

    @classmethod
    @abstractmethod
    def _resolve_type(cls, type_name: str) -> type[Self]:
        """Resolve type name into concrete subclass.

        Args:
            type_name: ``"module:Class"`` string.

        Returns:
            Concrete subclass type.
        """
        module, class_name = type_name.split(":", maxsplit=1)
        return getattr(import_module(module), class_name)

    @classmethod
    def resolve_type(cls, type_name: str) -> type[Self]:
        """Resolve type name into concrete subclass (public wrapper)."""
        return cls._resolve_type(type_name)

    @classmethod
    def auto(cls, settings: Settings | None = None) -> Self:
        """Build an instance based on TOML settings or defaults.

        Args:
            settings: Optional settings object to read from.

        Returns:
            The resolved instance.
        """
        if settings is None:
            settings = Settings()
        return cls._auto(settings=settings)

    @classmethod
    @cache
    def _auto(cls, settings: Settings) -> Self:
        """Internal cached auto-construction helper.

        Args:
            settings: Settings object.

        Returns:
            Resolved component instance.
        """
        key = cls._settings_key()

        if f"{key}_type" not in settings.toml_data:
            return cls._default()
        if not isinstance(settings.toml_data[f"{key}_type"], str):
            msg = f"Invalid type name for {key} in {settings.file}"
            raise TypeError(msg)

        class_type = cls._resolve_type(settings.toml_data[f"{key}_type"])
        class_kwargs = settings.toml_data.get(f"{key}_kwargs", {})
        if not isinstance(class_kwargs, dict):
            msg = f"Invalid kwargs for {key} in {settings.file}: expected table/dict."
            raise TypeError(msg)
        return class_type(**class_kwargs)

    def __reduce__(self) -> tuple[Callable[[type[msgspec.Struct], bytes], msgspec.Struct], tuple[type[Self], bytes]]:
        """Support pickling by reconstructing from msgpack bytes."""
        return (_reconstruct_struct, (type(self), msgspec.msgpack.encode(self)))


def _reconstruct_struct(cls: type[msgspec.Struct], serialized: bytes) -> msgspec.Struct:
    """Reconstruct msgspec struct from msgpack bytes."""
    return msgspec.msgpack.decode(serialized, type=cls)
