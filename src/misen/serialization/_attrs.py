import importlib.util
from typing import Any

_attrs_available = importlib.util.find_spec("attrs") is not None


def is_attrs_class(obj: Any) -> bool:
    if not _attrs_available:
        return False
    import attrs  # type: ignore

    return attrs.has(type(obj))
