from abc import ABC, abstractmethod
from typing import Any, Callable, cast

from misen_hash.utils import hash_msgspec

__all__ = ["canonical_hash", "PrimitiveHandler", "CollectionHandler"]


def canonical_hash(obj: Any) -> int:
    obj_type = type(obj).__qualname__
    obj_hash = _lookup_handler(obj).digest(obj, element_hash=canonical_hash)
    return hash_msgspec((obj_type, obj_hash))


## Handler ABCs


class Handler(ABC):
    @staticmethod
    @abstractmethod
    def match(obj: Any) -> bool: ...

    @staticmethod
    @abstractmethod
    def digest(obj: Any, element_hash: Callable[[Any], int] | None) -> int: ...


class PrimitiveHandler(Handler):
    @staticmethod
    @abstractmethod
    def digest(obj: Any, element_hash: None = None) -> int: ...


class CollectionHandler(Handler):
    @staticmethod
    @abstractmethod
    def elements(obj: Any) -> list[Any] | set[Any]: ...

    @classmethod
    def digest(cls, obj: Any, element_hash: Callable[[Any], int]) -> int:
        elements = cls.elements(obj)
        if isinstance(elements, list):
            return hash_msgspec([element_hash(i) for i in elements])
        elif isinstance(elements, set):
            return hash_msgspec({element_hash(i) for i in elements})
        raise ValueError(f"Unsupported collection type: {type(elements)}")


## Handlers by object type

from misen_hash._builtins import builtin_handlers, builtin_handlers_by_type  # noqa: E402
from misen_hash._dill import DillHandler  # noqa: E402
from misen_hash._torch import TorchModuleHandler, TorchTensorHandler  # noqa: E402

_handlers_type_cache: dict[type[Any], Handler] = {**builtin_handlers_by_type}


def _lookup_handler(obj: Any) -> Handler:
    obj_type = type(obj)

    if obj_type not in _handlers_type_cache:
        for hash_cls in cast("list[Handler]", builtin_handlers + [TorchTensorHandler, TorchModuleHandler]):
            if hash_cls.match(obj):
                _handlers_type_cache[obj_type] = hash_cls
                break
        else:
            _handlers_type_cache[obj_type] = DillHandler

    return _handlers_type_cache[obj_type]
