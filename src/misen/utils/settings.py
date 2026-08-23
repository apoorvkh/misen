"""Settings and singleton utilities for configurable components.

This module supports ``"auto"`` construction of executor/workspace instances
from layered TOML configuration and memoizes struct instances by constructor
kwargs for lightweight singleton behavior.

Config resolution order (lowest to highest priority):
1. ``$XDG_CONFIG_HOME/misen.toml`` — user-level defaults
2. ``./.misen.toml`` — project-level overrides (merged on top)
3. ``$MISEN_CONFIG`` env var or ``Settings(config_file=...)`` — explicit
   override that **replaces** the entire chain.
"""

import inspect
import os
import tomllib
import weakref
from abc import ABCMeta
from collections.abc import Callable, Mapping
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
        try:
            project_config = Path.cwd() / ".misen.toml"
        except OSError as exc:
            msg = f"Could not resolve Misen settings from the current working directory: {exc}"
            raise ConfigError(msg) from exc

        return (
            xdg_config_home / "misen.toml",
            project_config,
        )

    @cached_property
    def toml_data(self) -> dict[str, Any]:
        """Return parsed and merged TOML settings data.

        Raises:
            ConfigError: If a settings file cannot be read or parsed.
        """
        merged: dict[str, Any] = {}
        for path in self._config_files:
            try:
                raw_toml = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            except (OSError, UnicodeError) as exc:
                msg = f"Could not read Misen settings from {path}: {exc}"
                raise ConfigError(msg) from exc
            try:
                merged |= tomllib.loads(raw_toml)
            except tomllib.TOMLDecodeError as exc:
                msg = f"Invalid TOML in Misen settings file {path}: {exc}"
                raise ConfigError(msg) from exc
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
        key = msgspec.json.encode((cls.__module__, cls.__qualname__, kwargs))
        if (instance := ConfigurableMeta._instances.get(key)) is None:
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
    _config_validation_errors: ClassVar[tuple[type[Exception], ...]] = ()

    @classmethod
    def resolve_type(cls, type_name: str) -> type[Self]:
        """Resolve *type_name* to a concrete subclass.

        Checks ``_config_aliases`` first, then falls back to importing a
        ``"module:Class"`` string directly.

        Raises:
            ConfigError: If the reference is invalid, cannot be imported, or
                does not identify a subclass of this configurable type.
        """
        target = cls._config_aliases.get(type_name, type_name)
        module_name, separator, class_name = target.partition(":")
        if not separator or not module_name or not class_name:
            exc = ValueError(target)
            msg = f"Invalid {cls._config_key} type {target!r}; expected 'module:Class'."
            raise ConfigError(msg) from exc
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as exc:
            missing_name = exc.name or ""
            if missing_name != module_name and not module_name.startswith(f"{missing_name}."):
                raise
            msg = f"Could not resolve {cls._config_key} type {target!r}: {exc}"
            raise ConfigError(msg) from exc
        try:
            resolved = getattr(module, class_name)
        except AttributeError as exc:
            msg = f"Could not resolve {cls._config_key} type {target!r}: {exc}"
            raise ConfigError(msg) from exc
        if not isinstance(resolved, type) or not issubclass(resolved, cls):
            msg = f"Configured {cls._config_key} type {target!r} is not a {cls.__name__} subclass."
            raise ConfigError(msg)
        return resolved

    @classmethod
    def auto(cls, settings: Settings | None = None) -> Self:
        """Build an instance based on TOML settings or defaults.

        Args:
            settings: Optional settings object to read from.

        Returns:
            The resolved instance.

        Raises:
            ConfigError: If configuration cannot be parsed, converted,
                validated, or used to construct the configured component.
        """
        settings = Settings() if settings is None else settings
        raw_section = settings.toml_data.get(cls._config_key, {})
        if not isinstance(raw_section, Mapping):
            msg = f"Invalid [{cls._config_key}] settings: expected a TOML table."
            raise ConfigError(msg)
        section = dict(raw_section)
        type_name = section.pop("type", cls._config_default_type)
        if not isinstance(type_name, str):
            msg = f"Invalid type for [{cls._config_key}] in settings: expected string."
            raise ConfigError(msg)
        resolved_type = cls.resolve_type(type_name)
        constructor_signature = inspect.signature(resolved_type)
        try:
            constructor_signature.bind(**section)
            converted_section = {
                name: (
                    value
                    if (annotation := constructor_signature.parameters[name].annotation) is inspect.Parameter.empty
                    else msgspec.convert(value, type=annotation, strict=True)
                )
                for name, value in section.items()
            }
        except (TypeError, msgspec.ValidationError) as exc:
            msg = f"Invalid [{cls._config_key}] settings for {resolved_type.__name__}: {exc}"
            raise ConfigError(msg) from exc

        validation_errors = (msgspec.ValidationError, *getattr(resolved_type, "_config_validation_errors", ()))
        try:
            return resolved_type(**converted_section)
        except ConfigError:
            raise
        except validation_errors as exc:
            msg = f"Invalid [{cls._config_key}] settings for {resolved_type.__name__}: {exc}"
            raise ConfigError(msg) from exc

    @classmethod
    def resolve_auto(cls, /, obj: Self | Literal["auto"] = "auto") -> Self:
        """Resolve ``"auto"`` value."""
        return cls.auto() if obj == "auto" else obj

    def __reduce__(self) -> tuple[Callable[[type[msgspec.Struct], bytes], msgspec.Struct], tuple[type[Self], bytes]]:
        """Support pickling by reconstructing from msgpack bytes."""
        return (_reconstruct_struct, (type(self), msgspec.msgpack.encode(self)))


def _reconstruct_struct(cls: type[msgspec.Struct], serialized: bytes) -> msgspec.Struct:
    """Reconstruct msgspec struct from msgpack bytes."""
    return msgspec.msgpack.decode(serialized, type=cls)
