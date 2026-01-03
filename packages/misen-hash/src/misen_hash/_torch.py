import importlib.util
from typing import Any

_torch_available = importlib.util.find_spec("torch") is not None
_numpy_available = importlib.util.find_spec("numpy") is not None

__all__ = ["normalize_torch_object", "is_torch_object"]


def is_torch_object(obj: Any) -> bool:
    if _torch_available and type(obj).__module__.split(".")[0] == "torch":
        try:
            import torch
        except ImportError:
            return False

        if isinstance(obj, (torch.Tensor, torch.nn.Module, torch.optim.Optimizer)):
            if not _numpy_available:
                raise ImportError(
                    "numpy must be installed if using torch objects in misen. Please `pip install numpy`."
                )

            return True

    return False


def normalize_torch_object(obj) -> dict | tuple:
    import torch

    if isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)):
        return obj.state_dict()

    if isinstance(obj, torch.Tensor):
        t = obj.detach()
        if not t.is_contiguous():
            t = t.contiguous()
        if t.device.type != "cpu":
            t = t.cpu()

        # TODO: maybe just return this as bytes (without copy)? And have xxhash hash bytes directly instead of msgspec.encode bytes?

        # dtype, ndim, shape, raw tensor bytes (C-order, dtype-agnostic)
        return (str(t.dtype), t.ndim, (int(d) for d in t.shape), memoryview(t.view(torch.uint8).numpy()))

    raise ValueError(f"Unsupported torch object type: {type(obj)}")
