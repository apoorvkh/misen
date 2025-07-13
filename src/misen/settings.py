import builtins
import os
import sys
from abc import ABC, abstractmethod
from functools import cached_property
from importlib import import_module
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, cast

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


ConfigT = TypeVar("ConfigT", bound="ConfigABC")


class ConfigurableABC(Generic[ConfigT], ABC):
    @staticmethod
    @abstractmethod
    def config_type() -> type[ConfigT]:
        raise NotImplementedError

    def __init__(self, config: ConfigT):
        self.config = config


ConfigT = TypeVar("ConfigT", bound="ConfigABC")
ConfigurableT = TypeVar("ConfigurableT", bound="ConfigurableABC")


class ConfigABC(Struct, Generic[ConfigT, ConfigurableT], kw_only=True):
    type: str | Literal["auto"] = "auto"

    @staticmethod
    @abstractmethod
    def default() -> ConfigT:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def settings_key() -> str:
        raise NotImplementedError

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> ConfigT:
        if settings is None:
            settings = Settings()
        config_data = settings.toml_data.get(cls.settings_key(), None)
        if config_data is not None:
            config = msgspec.convert(config_data, type=cls)
            return msgspec.convert(config_data, type=config.resolve_config_type())
        return cls.default()

    def resolve_config_type(self) -> builtins.type[ConfigT]:
        return self.resolve_component_type().config_type()

    def resolve_component_type(self) -> builtins.type[ConfigurableT]:
        match self.type:
            case "auto":
                return self.from_settings().resolve_component_type()

        module, class_name = self.type.split(":", maxsplit=1)
        return getattr(import_module(module), class_name)

    def load(self, settings: Settings | None = None) -> ConfigurableT:
        match self.type:
            case "auto":
                config = self.from_settings(settings=settings)
            case _:
                config = self
        component_cls = config.resolve_component_type()
        return cast("ConfigurableT", component_cls(config=config))
