# import importlib.util
# from typing import Any

# _attrs_available = importlib.util.find_spec("attrs") is not None


# def is_attrs_class(obj: Any) -> bool:
#     if not _attrs_available:
#         return False

#     import attrs

#     return attrs.has(type(obj))


# def attrs_dictview(attrs_obj) -> dict[str, Any]:
#     from attrs import fields

#     return {f.name: getattr(attrs_obj, f.name) for f in fields(attrs_obj)}


# from typing import Any


# def is_msgspec_struct(obj: Any) -> bool:
#     import msgspec

#     return isinstance(obj, msgspec.Struct)


# def msgspec_struct_dictview(struct_obj) -> dict[str, Any]:
#     return {f: getattr(struct_obj, f) for f in struct_obj.__struct_fields__}


# class _HashWriter(io.RawIOBase):
#     """File-like sink that updates an xxhash object with whatever is written."""

#     def __init__(self) -> None:
#         self._h = xxh3_64(seed=0)

#     def writable(self) -> bool:
#         return True

#     def write(self, b: bytes | bytearray | memoryview) -> int:
#         mv = memoryview(b)
#         self._h.update(mv)
#         return mv.nbytes

#     def intdigest(self) -> int:
#         return self._h.intdigest()


# hash_sink = _HashWriter()
# dill.dump(obj, hash_sink, protocol=5)
# return hash_sink.intdigest()
