from __future__ import annotations

import inspect
import os
import sys
from abc import ABC, ABCMeta, abstractmethod
from functools import cache
from types import UnionType
from typing import Any, TypeAlias, Union, get_args, get_origin, get_type_hints

from typing_extensions import Self

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


ParamT: TypeAlias = str | int | float | bool | list["ParamT"] | dict[str, "ParamT"]


def _is_valid_annotation(annotation: type) -> bool:
    """If `annotation` is ParamT or Union[ParamT, ...]"""
    if annotation in (str, int, float, bool, ParamT):
        return True

    origin = get_origin(annotation)
    args = get_args(annotation)

    # a Union[T1, T2, ...]
    if origin is Union or origin is UnionType:
        return all(_is_valid_annotation(a) for a in args)

    # list[T]
    if origin is list and len(args) == 1:
        return _is_valid_annotation(args[0])

    # dict[str, T]
    if origin is dict and len(args) == 2 and args[0] is str:
        return _is_valid_annotation(args[1])

    return False


def _is_valid_value(val) -> bool:
    """Recursively check at runtime that `val` is of type ParamT."""
    if isinstance(val, (str, int, float, bool)):
        return True
    if isinstance(val, list):
        return all(_is_valid_value(v) for v in val)
    if isinstance(val, dict):
        return all(isinstance(k, str) and _is_valid_value(v) for k, v in val.items())
    return False


class FromParamsMeta(ABCMeta):
    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]):
        cls = super().__new__(mcs, name, bases, namespace)

        init_fn = namespace.get("__init__")

        if init_fn is None:
            return cls

        sig = inspect.signature(init_fn)
        params = list(sig.parameters.values())

        # inspect parameters besides 'self'
        if len(params) > 1:
            for p in params[1:]:
                if (
                    p.kind is not inspect.Parameter.KEYWORD_ONLY
                    and p.kind is not inspect.Parameter.VAR_KEYWORD
                ):
                    raise TypeError(
                        f"{name}.__init__ parameter '{p.name}' must be keyword-only"
                    )

            # fetch real annotations (resolving forward‑refs) for each param
            hints = get_type_hints(init_fn, globalns=init_fn.__globals__)

            # check that each parameter is valid (of type ParamT or Union[ParamT, ...])
            for p in params[1:]:
                annotation = hints.get(p.name, inspect._empty)
                if annotation is not inspect._empty and not _is_valid_annotation(annotation):
                    raise TypeError(
                        f"{name}.__init__ parameter '{p.name}' has invalid annotation {annotation!r}"
                    )

        return cls

    def __call__(cls, *args, **kwargs):
        if args:
            raise TypeError(f"{cls.__name__}() only accepts keyword arguments")

        # runtime‑type‑check every kwarg value
        for k, v in kwargs.items():
            if not _is_valid_value(v):
                raise TypeError(
                    f"{cls.__name__} argument '{k}' has invalid type {type(v).__name__}"
                )

        return super().__call__(**kwargs)


class FromParamsABC(ABC, metaclass=FromParamsMeta):
    """
    Subclasses must override __init__ with keyword-only params
    whose annotations are each unannotated or one of the allowed ArgT cases.
    """

    @abstractmethod
    def __init__(self, **kwargs: dict[str, ParamT]) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_params(cls, params: dict[str, ParamT] | None = None) -> Self:
        """Load a class from a dictionary of parameters."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def settings_key() -> str:
        """The key used in the TOML file to identify this class."""
        raise NotImplementedError

    @classmethod
    def from_settings(cls, path: str) -> Self:
        """Load a class from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)
        class_settings: dict[str, ParamT] | None = data.get(cls.settings_key(), None)
        return cls.from_params(class_settings)

    @classmethod
    @cache
    def default(cls) -> Self:
        """Load a class from the default settings."""
        return cls.from_settings(path=os.environ.get("MISEN_SETTINGS_PATH", "./settings.toml"))
