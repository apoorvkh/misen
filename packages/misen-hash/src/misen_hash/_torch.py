import importlib.util
from typing import Any, cast

from . import PrimitiveHandler
from .utils import hash_msgpack

_torch_available = importlib.util.find_spec("torch") is not None
_numpy_available = importlib.util.find_spec("numpy") is not None

__all__ = ["TorchTensorHandler", "TorchModuleHandler"]


class TorchTensorHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        if not _torch_available or type(obj).__module__.split(".")[0] != "torch":
            return False
        import torch

        return isinstance(obj, torch.Tensor)

    @staticmethod
    def digest(obj: Any, element_hash: None = None) -> int:
        if not _numpy_available:
            raise ImportError("numpy must be installed if using torch objects in misen. Please `pip install numpy`.")

        import torch

        t = cast("torch.Tensor", obj)
        t = t.detach()
        if not t.is_contiguous():
            t = t.contiguous()
        if t.device.type != "cpu":
            t = t.cpu()

        # TODO: maybe just return this as bytes (without copy)? And have xxhash hash bytes directly instead of msgspec.encode bytes?

        # dtype, ndim, shape, raw tensor bytes (C-order, dtype-agnostic)
        return hash_msgpack((str(t.dtype), t.ndim, (int(d) for d in t.shape), memoryview(t.view(torch.uint8).numpy())))


class TorchModuleHandler(PrimitiveHandler):
    @staticmethod
    def match(obj: Any) -> bool:
        if not _torch_available or type(obj).__module__.split(".")[0] != "torch":
            return False
        import torch

        return isinstance(obj, (torch.nn.Module, torch.optim.Optimizer))

    @staticmethod
    def elements(obj: Any) -> set[Any]:
        state_dict = obj.state_dict()
        if isinstance(state_dict, dict):
            return set(state_dict.items())
        return {state_dict}  # buffer
