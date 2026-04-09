"""Handlers for declarative torch config values (dtype, device)."""

import importlib.util
from typing import Any

from misen_hash.handler_base import HandlerTypeList, HandlerTypeRegistry, PrimitiveHandler
from misen_hash.hash import canonical_hash

__all__ = ["torch_handlers", "torch_handlers_by_type"]

torch_handlers: HandlerTypeList = []
torch_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("torch") is not None:

    class TorchDTypeHandler(PrimitiveHandler):
        """Hash torch dtype objects (e.g. torch.float32) by their stable string name."""

        @staticmethod
        def match(obj: Any) -> bool:
            import torch

            return isinstance(obj, torch.dtype)

        @staticmethod
        def digest(obj: Any) -> int:
            return canonical_hash(str(obj))

    class TorchDeviceHandler(PrimitiveHandler):
        """Hash torch device objects (e.g. torch.device('cuda:0')) by their string representation."""

        @staticmethod
        def match(obj: Any) -> bool:
            import torch

            return isinstance(obj, torch.device)

        @staticmethod
        def digest(obj: Any) -> int:
            return canonical_hash(str(obj))

    torch_handlers = [TorchDTypeHandler, TorchDeviceHandler]
    torch_handlers_by_type = {
        "torch.dtype": TorchDTypeHandler,
        "torch.device": TorchDeviceHandler,
    }
