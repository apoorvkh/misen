import os
import sys
from abc import ABC, abstractmethod
from functools import cached_property
from importlib import import_module
from pathlib import Path
from typing import Any, Generic, Type, TypeVar, cast

import msgspec
from msgspec import Struct

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class Settings(Struct, dict=True):
    file: Path = Path(os.environ.get("MISEN_SETTINGS_FILE") or (Path.cwd() / "misen_settings.toml"))

    @cached_property
    def toml_data(self) -> dict[str, Any]:
        try:
            return tomllib.loads(self.file.read_bytes().decode())
        except FileNotFoundError:
            return {}


C = TypeVar("C", bound="ConfigABC")


class TargetABC(Generic[C], ABC):
    @staticmethod
    @abstractmethod
    def config_type() -> Type[C]:
        raise NotImplementedError

    def __init__(self, config: C):
        self.config = config


C = TypeVar("C", bound="ConfigABC")
T = TypeVar("T", bound="TargetABC")


class ConfigABC(Struct, Generic[C, T], kw_only=True):
    type: str | None = None

    @abstractmethod
    def default(self) -> C:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def settings_key() -> str:
        raise NotImplementedError

    def from_settings(self, settings: Settings | None = None) -> C:
        if settings is None:
            settings = Settings()
        config_data = settings.toml_data.get(self.settings_key(), None)
        if config_data is not None:
            config = msgspec.convert(config_data, type=type(self))
            return msgspec.convert(config_data, type=config.resolve_config_type())
        return self.default()

    def resolve_config_type(self) -> Type[C]:
        return self.resolve_target_type().config_type()

    def resolve_target_type(self) -> Type[T]:
        if self.type is None:
            return self.from_settings().resolve_target_type()

        module, class_name = self.type.split(":", maxsplit=1)
        return getattr(import_module(module), class_name)

    def load_target(self, settings: Settings | None = None) -> T:
        if self.type is None:
            config = self.from_settings(settings=settings)
        else:
            config = self
        target_cls = config.resolve_target_type()
        return cast("T", target_cls(config=config))
