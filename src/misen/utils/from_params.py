from __future__ import annotations

import os
from abc import abstractmethod
from functools import cache

import msgspec
from typing_extensions import Self


class FromParamsABC(msgspec.Struct):
    @classmethod
    def _from_params(cls, params: dict) -> Self:
        return msgspec.convert(params, cls)

    @classmethod
    @abstractmethod
    def from_params(cls, params: dict) -> Self:
        return cls._from_params(params)

    @classmethod
    @abstractmethod
    def default_params(cls) -> dict:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def toml_key(cls) -> str:
        raise NotImplementedError

    @classmethod
    def from_toml(cls, p: str | os.PathLike) -> Self:
        toml_data = msgspec.toml.decode(open(p, "r").read())
        params = toml_data.get(cls.toml_key(), cls.default_params())
        return cls.from_params(params=params)

    @classmethod
    @cache
    def default(cls) -> Self:
        print("Hello")
        settings_file = os.environ.get("SETTINGS_FILE", "settings.toml")
        if os.path.exists(settings_file):
            return cls.from_toml(settings_file)
        return cls.from_params(params=cls.default_params())
