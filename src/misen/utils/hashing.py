import datetime
import decimal
import enum
import importlib.util
import sys
import uuid
from dataclasses import is_dataclass
from typing import Any, Callable, Type

import dill
import msgspec
import msgspec.json
from xxhash import xxh3_64_intdigest


def deterministic_hash(obj: Any, seed: int = 0) -> int:
    return xxh3_64_intdigest(serialize(obj), seed=seed)


def serialize(obj: Any) -> bytes:
    obj_type = type(obj)

    if obj_type in _SERIALIZERS:
        serializer = _SERIALIZERS[obj_type]
    elif isinstance(obj, msgspec.Struct) or is_dataclass(obj) or _is_attrs_class(obj):
        serializer = _msgspec_serializer
    elif _is_torch_object(obj):
        serializer = _torch_serializer
    elif hasattr(obj, "__getstate__"):
        serializer = lambda o: serialize(o.__getstate__())  # noqa: E731
    elif hasattr(obj, "__dict__"):
        serializer = lambda o: serialize(o.__dict__)  # noqa: E731
    elif hasattr(obj, "__slots__"):
        serializer = lambda o: serialize(  # noqa: E731
            {slot: getattr(o, slot) for slot in o.__slots__}
        )
    else:
        # TODO: add logger.warning to flag fallback
        serializer = dill.dumps

    type_name = obj_type.__qualname__.encode("utf-8")
    payload = serializer(obj)
    return _msgspec_serializer((type_name, payload))


_SERIALIZERS: dict[Type, Callable[[Any], bytes]] = {}


def register_serializer(types: list[Type], fn: Callable[[Any], bytes]) -> None:
    for typ in types:
        _SERIALIZERS[typ] = fn


def _msgspec_serializer(obj: Any) -> bytes:
    return msgspec.json.encode(obj, enc_hook=serialize, order="sorted")


register_serializer(
    [type(None), bool, int, float, str, bytes, bytearray], _msgspec_serializer
)

register_serializer(
    [list, tuple, set, frozenset],
    lambda o: _msgspec_serializer([serialize(i) for i in o]),
)
register_serializer(
    [dict],
    lambda o: _msgspec_serializer(
        {serialize(k).decode(encoding="latin-1"): serialize(v) for k, v in o.items()}
    ),
)

register_serializer(
    [
        datetime.datetime,
        datetime.date,
        datetime.time,
        datetime.timedelta,
        uuid.UUID,
        decimal.Decimal,
        enum.Enum,
        enum.IntEnum,
        enum.Flag,
        enum.IntFlag,
    ]
    + ([enum.StrEnum] if sys.version_info >= (3, 11) else []),
    _msgspec_serializer,
)

_attrs_available = importlib.util.find_spec("attrs") is not None


def _is_attrs_class(obj: Any) -> bool:
    if not _attrs_available:
        return False
    import attrs  # type: ignore

    return attrs.has(type(obj))


_torch_available = importlib.util.find_spec("torch") is not None


def _is_torch_object(obj: Any) -> bool:
    if _torch_available and type(obj).__module__.split(".")[0] == "torch":
        try:
            import torch

            return isinstance(
                obj, (torch.Tensor, torch.nn.Module, torch.optim.Optimizer)
            )
        except ImportError:
            return False
    return False


def _torch_serializer(obj) -> bytes:
    import io

    import torch

    if isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)):
        return serialize(obj.state_dict())

    t: torch.Tensor = obj
    buffer = io.BytesIO()
    torch.save(t.detach().cpu().contiguous(), buffer)
    return buffer.getvalue()
