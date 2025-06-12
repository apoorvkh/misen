from typing import Any


def is_msgspec_struct(obj: Any) -> bool:
    import msgspec

    return isinstance(obj, msgspec.Struct)
