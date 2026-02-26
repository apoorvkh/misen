"""Optional torch-aware handlers for stable tensor/module hashing."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import hash_msgspec, incremental_hash

__all__ = ["torch_handlers", "torch_handlers_by_type"]


torch_handlers: HandlerTypeList = []
torch_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("torch") is not None:
    _numpy_available = importlib.util.find_spec("numpy") is not None

    def _has_torch_base(obj: Any, type_name: str) -> bool:
        return any(
            base.__name__ == type_name and base.__module__.split(".")[0] == "torch"
            for base in type(obj).__mro__
        )

    class TorchTensorHandler(PrimitiveHandler):
        """Hash tensor contents (dtype/shape/bytes) independent of device."""

        @staticmethod
        def match(obj: Any) -> bool:
            """Return True for torch Tensor instances."""
            return _has_torch_base(obj, "Tensor")

        @staticmethod
        def digest(obj: Any) -> int:
            """Hash tensor metadata plus raw byte payload in contiguous CPU form."""
            if not _numpy_available:
                msg = "numpy must be installed if using torch objects in misen. Please `pip install numpy`."
                raise ImportError(msg)

            t = obj
            t = t.detach()
            if not t.is_contiguous():
                t = t.contiguous()
            if t.device.type != "cpu":
                t = t.cpu()

            payload_hash = incremental_hash(lambda sink: sink.write(memoryview(t.numpy().view("uint8"))))

            # Tensor identity for caching: dtype + rank + shape + raw bytes.
            return hash_msgspec(
                (
                    str(t.dtype),
                    t.ndim,
                    tuple(int(d) for d in t.shape),
                    payload_hash,
                )
            )


    class TorchModuleHandler(CollectionHandler):
        """Hash module/optimizer state dictionaries as element collections."""

        @staticmethod
        def match(obj: Any) -> bool:
            """Return True for torch Module and Optimizer instances."""
            return _has_torch_base(obj, "Module") or _has_torch_base(obj, "Optimizer")

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            """Expose state-dict entries as a sorted list for deterministic hashing."""
            state_dict = obj.state_dict()
            if isinstance(state_dict, dict):
                sorted_keys = sorted(
                    state_dict,
                    key=lambda key: (type(key).__module__, type(key).__qualname__, repr(key)),
                )
                return [(key, state_dict[key]) for key in sorted_keys]
            return [state_dict]

    torch_handlers = [TorchTensorHandler, TorchModuleHandler]
    torch_handlers_by_type = {
        "torch.Tensor": TorchTensorHandler,
        "torch.nn.modules.module.Module": TorchModuleHandler,
        "torch.optim.optimizer.Optimizer": TorchModuleHandler,
    }
