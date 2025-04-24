from __future__ import annotations

import os
from abc import abstractmethod
from datetime import date, datetime, time
from functools import cache

import msgspec
from typing_extensions import Self

TomlPrimitive = str | int | float | bool | datetime | date | time
TomlType = TomlPrimitive | list["TomlType"] | dict[str, "TomlType"]


class FromParamsABC(msgspec.Struct):
    @classmethod
    def _from_params(cls, params: dict[str, TomlType]) -> Self:
        return msgspec.convert(params, cls)

    @classmethod
    @abstractmethod
    def from_params(cls, params: dict[str, TomlType]) -> Self:
        return cls._from_params(params)

    @classmethod
    @abstractmethod
    def default_params(cls) -> dict[str, TomlType]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def toml_key(cls) -> str:
        raise NotImplementedError

    @classmethod
    def from_toml(cls, p: str | os.PathLike) -> Self:
        toml_data: dict[str, TomlType] = msgspec.toml.decode(open(p, "r").read())
        params = toml_data.get(cls.toml_key(), cls.default_params())
        if not isinstance(params, dict):
            raise ValueError(f'Expected a dictionary for "{cls.toml_key()}" key in {p}.')
        return cls.from_params(params=params)

    @classmethod
    @cache
    def default(cls) -> Self:
        settings_file = os.environ.get("SETTINGS_FILE", "settings.toml")
        if os.path.exists(settings_file):
            return cls.from_toml(settings_file)
        return cls.from_params(params=cls.default_params())
