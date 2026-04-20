"""Serializers for PyTorch tensors, tensor dicts, and ``nn.Module`` models."""

import importlib.util
from collections import OrderedDict
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
        """Serialize a single ``torch.Tensor`` via ``torch.save``.

        Loaded with ``torch.load(weights_only=True)`` which restricts pickle
        execution to a tensor-safe allowlist of globals.  ``.detach().cpu()``
        on write makes the save portable to CPU-only machines.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            import torch

            return isinstance(obj, torch.Tensor)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import torch

            torch.save(obj.detach().cpu(), directory / "data.pt")
            return {
                "torch_version": torch.__version__,
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import torch

            return torch.load(directory / "data.pt", weights_only=True, map_location="cpu")

    class DictOfTensorsSerializer(Serializer[Any]):
        """Serialize ``dict[str, torch.Tensor]`` / ``OrderedDict[str, torch.Tensor]``.

        Targets state_dicts and flat tensor collections.  Requires all keys
        to be ``str`` and all values to be ``torch.Tensor`` — mixed dicts
        fall through to :class:`MsgpackSerializer` (which will then raise
        :class:`UnserializableTypeError` for the tensor values).

        Uses ``torch.save`` on the whole dict and ``torch.load(weights_only=True)``
        on load.  Pickle preserves the container type (``dict`` vs ``OrderedDict``)
        and insertion order natively, so no extra metadata is needed beyond the
        torch version.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            # Strict type match — subclasses like ``defaultdict`` fall through.
            if type(obj) is not dict and type(obj) is not OrderedDict:
                return False
            if not obj:
                # Empty dict — let MsgpackSerializer handle it (cheaper, no
                # binary file written for a trivial value).
                return False
            if not all(isinstance(k, str) for k in obj):
                return False
            import torch

            return all(isinstance(v, torch.Tensor) for v in obj.values())

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import torch

            # Build a cpu-detached copy while preserving container type
            # (``type(obj)(iterable_of_pairs)`` works for both dict and
            # OrderedDict).  The container type then rides through pickle.
            cpu_dict = type(obj)((k, v.detach().cpu()) for k, v in obj.items())
            torch.save(cpu_dict, directory / "data.pt")
            return {"torch_version": torch.__version__}

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

    # Order within this list does not affect correctness — the three match
    # methods are mutually exclusive (Tensor vs. dict-of-Tensor vs. Module),
    # and single-tensor / Module dispatch goes through the by-type fast
    # path below anyway.  Order reflects reading convenience only.
    torch_serializers = [TorchTensorSerializer, DictOfTensorsSerializer, TorchModuleSerializer]
    torch_serializers_by_type = {
        "torch.Tensor": TorchTensorSerializer,
        "torch.nn.modules.module.Module": TorchModuleSerializer,
        # NOTE: DictOfTensorsSerializer is intentionally NOT listed here
        # because ``dict`` / ``OrderedDict`` are registered as volatile_types
        # on the serde registry — dispatch happens via the linear-scan
        # ``match`` path so the content check runs every call.
    }
