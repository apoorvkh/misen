import importlib.util
from typing import Any, cast

from . import CollectionForHashing, HashPrimitive, HashTree

_torch_available = importlib.util.find_spec("torch") is not None

__all__ = ["TensorPrimitive"]


class TensorPrimitive(HashPrimitive):
    @staticmethod
    def match(obj: Any) -> bool:
        if not _torch_available or type(obj).__module__.split(".")[0] != "torch":
            return False
        import torch

        return isinstance(obj, torch.Tensor)

    @staticmethod
    def digest(obj) -> int:
        import torch

        obj = cast(torch.Tensor, obj)
        return 0
