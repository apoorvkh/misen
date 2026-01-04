from typing import Any

from . import CollectionForHashing, HashPrimitive, HashTree


class DictCollection(CollectionForHashing):
    @staticmethod
    def match(obj: Any) -> bool:
        return isinstance(obj, dict)

    @staticmethod
    def children(obj: Any) -> set[Any]:
        return {i for kv in obj.items() for i in kv}
