"""Settings and singleton utilities for configurable components.

This module supports ``"auto"`` construction of executor/workspace instances
from layered TOML configuration and memoizes struct instances by constructor
kwargs for lightweight singleton behavior.

Config resolution order (lowest to highest priority):
1. ``$XDG_CONFIG_HOME/misen/config.toml`` — user-level defaults
2. ``./.misen.toml`` — project-level overrides (merged on top)
3. ``$MISEN_CONFIG`` env var or ``Settings(config_file=...)`` — explicit
   override that **replaces** the entire chain.
"""

import os
import tomllib
import weakref
from abc import ABCMeta
from collections.abc import Callable
from functools import cached_property
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

import msgspec
from msgspec import Struct

from misen.exceptions import ConfigError

__all__ = ["Configurable", "Settings"]


def _file_stat_key(path: Path) -> tuple[str, int, int] | None:
    """Return ``(resolved_path, mtime_ns, size)`` or ``None`` if missing."""
    resolved = path.expanduser().resolve()
    try:
        stat = resolved.stat()
        return (str(resolved), stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        return None


class Settings(Struct, dict=True):
    """Layered TOML configuration loader.

    When *config_file* is ``None`` (the default), settings are resolved by
    merging ``$XDG_CONFIG_HOME/misen.toml`` with
    ``./.misen.toml``.  Project-level sections replace XDG sections entirely.
    The ``$MISEN_CONFIG`` environment variable, if set,
    short-circuits this and uses only that single file.  An explicit
    *config_file* argument behaves the same way.
    """

    config_file: Path | None = None

    @cached_property
    def _config_files(self) -> tuple[Path, ...]:
        """Return the ordered list of config files to load."""
        if self.config_file is not None:
            return (self.config_file,)

        if "MISEN_CONFIG" in os.environ:
            return (Path(os.environ["MISEN_CONFIG"]),)

        xdg_config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))

        return (
            xdg_config_home / "misen.toml",
            Path.cwd() / ".misen.toml",
        )

    @cached_property
    def toml_data(self) -> dict[str, Any]:
        """Return parsed and merged TOML settings data."""
        merged: dict[str, Any] = {}
        for path in self._config_files:
            if path.exists():
                merged |= tomllib.loads(path.read_bytes().decode())
        return merged

    def __hash__(self) -> int:
        """Return hash based on config file identities and stat metadata."""
        return hash(tuple(_file_stat_key(p) for p in self._config_files))


class ConfigurableMeta(msgspec.StructMeta, ABCMeta):
    """Metaclass implementing a parameterized singleton cache.

    Instances are cached in a ``WeakValueDictionary`` keyed by
    ``(module, qualname, kwargs)``.  Entries are automatically evicted when
    the last external strong reference to an instance is dropped, so the
    cache acts as a memoization layer for live instances rather than a
    persistent registry.  This requires ``Configurable`` to opt into weak
    reference support via ``weakref=True``.
    """

    _instances: ClassVar["weakref.WeakValueDictionary[bytes, Any]"] = weakref.WeakValueDictionary()

    def __call__(cls, **kwargs: Any) -> Any:
        """Return memoized instance for given constructor kwargs."""
        key = msgspec.json.encode((str(cls.__module__), str(cls.__qualname__), kwargs))
        instance = ConfigurableMeta._instances.get(key)
        if instance is None:
            instance = super().__call__(**kwargs)
            ConfigurableMeta._instances[key] = instance
        return instance


class Configurable(msgspec.Struct, dict=True, weakref=True, metaclass=ConfigurableMeta):
    """Base class for settings-backed singleton structs.

    Subclasses declare three ``ClassVar`` attributes instead of overriding
    abstract methods:

    - ``_config_key``: TOML section name (e.g. ``"workspace"``).
    - ``_config_default_type``: ``"module:Class"`` for the default implementation.
    - ``_config_aliases``: mapping of shorthand names to ``"module:Class"`` strings.
    """

    _config_key: ClassVar[str]
    _config_default_type: ClassVar[str]
    _config_aliases: ClassVar[dict[str, str]]

    @classmethod
    def resolve_type(cls, type_name: str) -> type[Self]:
        """Resolve *type_name* to a concrete subclass.

        Checks ``_config_aliases`` first, then falls back to importing a
        ``"module:Class"`` string directly.
        """
        target = cls._config_aliases.get(type_name, type_name)
        module, class_name = target.split(":", maxsplit=1)
        return getattr(import_module(module), class_name)

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

        section = dict(settings.toml_data.get(cls._config_key, {}))
        type_name = section.pop("type", None)

        if type_name is None:
            if section:
                return cls.resolve_type(cls._config_default_type)(**section)
            return cls.resolve_type(cls._config_default_type)()

        if not isinstance(type_name, str):
            msg = f"Invalid type for [{cls._config_key}] in settings: expected string."
            raise ConfigError(msg)

        return cls.resolve_type(type_name)(**section)

    @classmethod
    def resolve_auto(cls, /, obj: Self | Literal["auto"] = "auto") -> Self:
        """Resolve ``"auto"`` value."""
        if obj == "auto":
            return cls.auto()
        return obj

    def __reduce__(self) -> tuple[Callable[[type[msgspec.Struct], bytes], msgspec.Struct], tuple[type[Self], bytes]]:
        """Support pickling by reconstructing from msgpack bytes."""
        return (_reconstruct_struct, (type(self), msgspec.msgpack.encode(self)))


def _reconstruct_struct(cls: type[msgspec.Struct], serialized: bytes) -> msgspec.Struct:
    """Reconstruct msgspec struct from msgpack bytes."""
    return msgspec.msgpack.decode(serialized, type=cls)
