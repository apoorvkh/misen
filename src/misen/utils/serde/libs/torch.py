"""Serializers for PyTorch tensors and nn.Module models."""

import importlib.util
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import (
    Serializer,
    SerializerTypeRegistry,
)

__all__ = ["torch_serializers", "torch_serializers_by_type"]

torch_serializers: list[type[Serializer]] = []
torch_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("torch") is not None:
    from pathlib import Path

    class TorchTensorSerializer(Serializer[Any]):
        """Serialize ``torch.Tensor`` via ``torch.save`` with ``weights_only=True``."""

        @staticmethod
        def match(obj: Any) -> bool:
            import torch

            return isinstance(obj, torch.Tensor)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import torch

            torch.save(obj.detach().cpu(), directory / "data.pt", weights_only=True)
            return {
                "torch_version": torch.__version__,
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import torch

            return torch.load(directory / "data.pt", weights_only=True, map_location="cpu")

    class TorchModuleSerializer(Serializer[Any]):
        """Serialize ``torch.nn.Module`` via ``torch.save``.

        Uses ``torch.save(model, path)`` which saves the full model including
        its class structure.  Requires ``weights_only=False`` on load.

        Note: This uses pickle internally.  Serialized models may not load
        across different PyTorch versions.  The torch version is recorded in
        metadata for diagnostics.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            import torch.nn

            return isinstance(obj, torch.nn.Module)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import torch

            torch.save(obj, directory / "model.pt")
            return {"torch_version": torch.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import torch

            return torch.load(directory / "model.pt", weights_only=False, map_location="cpu")

    torch_serializers = [TorchTensorSerializer, TorchModuleSerializer]
    torch_serializers_by_type = {
        "torch.Tensor": TorchTensorSerializer,
        "torch.nn.modules.module.Module": TorchModuleSerializer,
    }
