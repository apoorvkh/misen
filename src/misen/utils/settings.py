import os
import sys
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from functools import cached_property
from importlib import import_module
from pathlib import Path
from typing import Any

import msgspec
from msgspec import Struct
from typing_extensions import Self

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


__all__ = ["Settings", "FromSettingsABC"]


DEFAULT_SETTINGS_FILE = Path(os.environ.get("MISEN_SETTINGS_FILE") or (Path.cwd() / "misen.toml"))


class Settings(Struct, dict=True):
    file: Path = DEFAULT_SETTINGS_FILE

    @cached_property
    def toml_data(self) -> dict[str, Any]:
        try:
            return tomllib.loads(self.file.read_bytes().decode())
        except FileNotFoundError:
            return {}


class FromSettingsMeta(msgspec.StructMeta, ABCMeta):
    _instances: dict[bytes, Any] = {}

    def __call__(cls, **kwargs):
        """Parameterized singleton"""
        key = msgspec.json.encode((str(cls.__module__), str(cls.__qualname__), kwargs))
        if key not in FromSettingsMeta._instances:
            FromSettingsMeta._instances[key] = super().__call__(**kwargs)
        return FromSettingsMeta._instances[key]


class FromSettingsABC(msgspec.Struct, dict=True, metaclass=FromSettingsMeta):
    @staticmethod
    @abstractmethod
    def _default() -> Self: ...

    @staticmethod
    @abstractmethod
    def _settings_key() -> str: ...

    @classmethod
    @abstractmethod
    def _resolve_type(cls, type_name: str) -> type[Self]:
        module, class_name = type_name.split(":", maxsplit=1)
        return getattr(import_module(module), class_name)

    @classmethod
    def auto(cls, settings: Settings | None = None) -> Self:
        if settings is None:
            settings = Settings()

        key = cls._settings_key()

        if f"{key}_type" not in settings.toml_data:
            return cls._default()

        class_type = cls._resolve_type(settings.toml_data.get[f"{key}_type"])
        class_kwargs = settings.toml_data.get(f"{key}_kwargs", {})
        return class_type(**class_kwargs)

    def __reduce__(self) -> tuple[Callable[[type[msgspec.Struct], bytes], msgspec.Struct], tuple[type[Self], bytes]]:
        return (_reconstruct_struct, (type(self), msgspec.msgpack.encode(self)))


def _reconstruct_struct(cls: type[msgspec.Struct], serialized: bytes) -> msgspec.Struct:
    return msgspec.msgpack.decode(serialized, type=cls)
