"""Optional torch-aware handlers for stable tensor/module hashing."""

import importlib.util
from typing import Any, cast

from misen_hash import CollectionHandler, PrimitiveHandler
from misen_hash.utils import hash_msgspec

_torch_available = importlib.util.find_spec("torch") is not None
_numpy_available = importlib.util.find_spec("numpy") is not None

__all__ = ["TorchModuleHandler", "TorchTensorHandler"]


class TorchTensorHandler(PrimitiveHandler):
    """Hash tensor contents (dtype/shape/bytes) independent of device."""

    @staticmethod
    def match(obj: Any) -> bool:
        if not _torch_available or type(obj).__module__.split(".")[0] != "torch":
            return False
        import torch

        return isinstance(obj, torch.Tensor)

    @staticmethod
    def digest(obj: Any, element_hash: None = None) -> int:
        """Hash tensor metadata plus raw byte payload in contiguous CPU form."""
        _ = element_hash
        if not _numpy_available:
            msg = "numpy must be installed if using torch objects in misen. Please `pip install numpy`."
            raise ImportError(msg)

        import torch

        t = cast("torch.Tensor", obj)
        t = t.detach()
        if not t.is_contiguous():
            t = t.contiguous()
        if t.device.type != "cpu":
            t = t.cpu()

        # Tensor identity for caching: dtype + rank + shape + raw bytes.
        return hash_msgspec(
            (
                str(t.dtype),
                t.ndim,
                tuple(int(d) for d in t.shape),
                memoryview(t.view(torch.uint8).numpy()),
            )
        )


class TorchModuleHandler(CollectionHandler):
    """Hash module/optimizer state dictionaries as element collections."""

    @staticmethod
    def match(obj: Any) -> bool:
        if not _torch_available or type(obj).__module__.split(".")[0] != "torch":
            return False
        import torch

        return isinstance(obj, (torch.nn.Module, torch.optim.Optimizer))

    @staticmethod
    def elements(obj: Any) -> set[Any]:
        """Expose state-dict entries as a set for order-insensitive hashing."""
        state_dict = obj.state_dict()
        if isinstance(state_dict, dict):
            return set(state_dict.items())
        return {state_dict}
