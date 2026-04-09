"""Serializer for PyTorch tensors."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["torch_serializers", "torch_serializers_by_type"]

torch_serializers: SerializerTypeList = []
torch_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("torch") is not None:
    from pathlib import Path

    class TorchTensorSerializer(Serializer[Any]):
        """Serialize ``torch.Tensor`` via ``torch.save`` with ``weights_only=True``."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import torch

            return isinstance(obj, torch.Tensor)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import torch

            torch.save(obj.detach().cpu(), directory / "data.pt", weights_only=True)
            write_meta(
                directory,
                TorchTensorSerializer,
                torch_version=torch.__version__,
                dtype=str(obj.dtype),
                shape=list(obj.shape),
            )

        @staticmethod
        def load(directory: Path) -> Any:
            import torch

            return torch.load(directory / "data.pt", weights_only=True, map_location="cpu")

    torch_serializers = [TorchTensorSerializer]
    torch_serializers_by_type = {"torch.Tensor": TorchTensorSerializer}
