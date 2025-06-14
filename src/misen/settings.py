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


ConfigT = TypeVar("ConfigT", bound="ConfigABC")


class ComponentABC(Generic[ConfigT], ABC):
    @staticmethod
    @abstractmethod
    def config_type() -> Type[ConfigT]:
        raise NotImplementedError

    def __init__(self, config: ConfigT):
        self.config = config


ConfigT = TypeVar("ConfigT", bound="ConfigABC")
ComponentT = TypeVar("ComponentT", bound="ComponentABC")


class ConfigABC(Struct, Generic[ConfigT, ComponentT], kw_only=True):
    type: str | None = None

    @abstractmethod
    def default(self) -> ConfigT:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def settings_key() -> str:
        raise NotImplementedError

    def from_settings(self, settings: Settings | None = None) -> ConfigT:
        if settings is None:
            settings = Settings()
        config_data = settings.toml_data.get(self.settings_key(), None)
        if config_data is not None:
            config = msgspec.convert(config_data, type=type(self))
            return msgspec.convert(config_data, type=config.resolve_config_type())
        return self.default()

    def resolve_config_type(self) -> Type[ConfigT]:
        return self.resolve_component_type().config_type()

    def resolve_component_type(self) -> Type[ComponentT]:
        if self.type is None:
            return self.from_settings().resolve_component_type()

        module, class_name = self.type.split(":", maxsplit=1)
        return getattr(import_module(module), class_name)

    def load(self, settings: Settings | None = None) -> ComponentT:
        if self.type is None:
            config = self.from_settings(settings=settings)
        else:
            config = self
        component_cls = config.resolve_component_type()
        return cast("ComponentT", component_cls(config=config))
