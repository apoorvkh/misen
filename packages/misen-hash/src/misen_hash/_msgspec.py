from typing import Any


def is_msgspec_struct(obj: Any) -> bool:
    import msgspec

    return isinstance(obj, msgspec.Struct)


def msgspec_struct_dictview(struct_obj) -> dict[str, Any]:
    return {f: getattr(struct_obj, f) for f in struct_obj.__struct_fields__}
