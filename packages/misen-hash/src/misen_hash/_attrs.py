import importlib.util
from typing import Any

_attrs_available = importlib.util.find_spec("attrs") is not None


def is_attrs_class(obj: Any) -> bool:
    if not _attrs_available:
        return False

    import attrs

    return attrs.has(type(obj))


def attrs_dictview(attrs_obj) -> dict[str, Any]:
    from attrs import fields

    return {f.name: getattr(attrs_obj, f.name) for f in fields(attrs_obj)}
