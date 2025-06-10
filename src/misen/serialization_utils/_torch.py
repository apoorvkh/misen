import importlib.util
from typing import Any

_torch_available = importlib.util.find_spec("torch") is not None


def is_torch_object(obj: Any) -> bool:
    if _torch_available and type(obj).__module__.split(".")[0] == "torch":
        try:
            import torch

            return isinstance(obj, (torch.Tensor, torch.nn.Module, torch.optim.Optimizer))
        except ImportError:
            return False
    return False


def torch_serializer(obj) -> bytes:
    import io

    import torch

    from . import serialize

    if isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)):
        return serialize(obj.state_dict())

    t: torch.Tensor = obj
    buffer = io.BytesIO()
    torch.save(t.detach().cpu().contiguous(), buffer)
    return buffer.getvalue()
