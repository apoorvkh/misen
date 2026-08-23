"""Torch serializers for v2.

- :class:`TorchTensorSerializer` — a :class:`LeafSerializer` that
  batches all tensors in a save into a single ``tensors.pt``.  This
  is the demonstration of the design's main win: a deeply nested dict
  of tensors (e.g. state_dict of state_dicts) packs into one
  ``torch.save`` call, matching the v1 flat case and extending it.
- :class:`TorchModuleSerializer` — a :class:`Serializer` that
  owns its own subdirectory (escape hatch for anything that doesn't
  fit the batching model).
"""

import importlib.util
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.exceptions import SerializationError
from misen.utils.serde.base import BaseSerializer, LeafSerializer, Serializer, translate_errors

__all__ = ["torch_serializers", "torch_serializers_by_type"]

torch_serializers: list[type[BaseSerializer]] = []
torch_serializers_by_type: dict[str, type[BaseSerializer]] = {}

if importlib.util.find_spec("torch") is not None:

    class TorchTensorSerializer(LeafSerializer[Any]):
        """Leaf serializer for ``torch.Tensor``.

        ``to_payload`` returns a CPU-detached copy so saves are
        portable.  ``write_batch`` packs all leaves of this kind into a
        single ``tensors.pt`` file keyed by ``leaf_id``; ``read_batch``
        opens the file once and returns a ``leaf_id → Tensor`` reader.
        """

        leaf_kind = "torch_tensor"

        @staticmethod
        def match(obj: Any) -> bool:
            import torch

            return isinstance(obj, torch.Tensor)

        @classmethod
        def to_payload(cls, obj: Any) -> Any:
            return obj.detach().cpu()

        @staticmethod
        def write_batch(
            entries: list[tuple[str, Any, Mapping[str, Any]]],
            directory: Path,
        ) -> Mapping[str, Any]:
            import torch

            bundle = {leaf_id: payload for leaf_id, payload, _ in entries}
            with translate_errors(
                f"Could not encode torch tensor bundle in {directory}",
                (OSError, RuntimeError, TypeError, pickle.PickleError),
            ):
                torch.save(bundle, directory / "tensors.pt")
            return {"torch_version": torch.__version__}

        @staticmethod
        def read_batch(directory: Path, kind_meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import torch

            with translate_errors(
                f"Could not decode torch tensor bundle in {directory}",
                (OSError, EOFError, RuntimeError, TypeError, ValueError, pickle.PickleError),
            ):
                bundle = torch.load(directory / "tensors.pt", weights_only=True, map_location="cpu")

            def reader(leaf_id: str) -> Any:
                try:
                    return bundle[leaf_id]
                except (KeyError, TypeError) as exc:
                    msg = f"Torch tensor bundle in {directory} does not contain leaf {leaf_id!r}."
                    raise SerializationError(msg) from exc

            return reader

    class TorchModuleSerializer(Serializer[Any]):
        """Directory-owning serializer for ``torch.nn.Module``.

        Modules don't fit the leaf-batching model (each module has
        bespoke structure), so this falls back to the escape hatch:
        ``write`` writes ``model.pt`` in its own subdir and
        ``read`` reads it back.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            import torch.nn

            return isinstance(obj, torch.nn.Module)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import torch

            with translate_errors(
                f"Could not encode torch module in {directory}",
                (OSError, RuntimeError, TypeError, pickle.PickleError),
            ):
                torch.save(obj, directory / "model.pt")
            return {"torch_version": torch.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import torch

            with translate_errors(
                f"Could not decode torch module in {directory}",
                (OSError, EOFError, RuntimeError, TypeError, ValueError, pickle.PickleError),
            ):
                return torch.load(directory / "model.pt", weights_only=False, map_location="cpu")

    # Tensor match must come before Module match? No — they're mutually
    # exclusive (Tensor isn't an nn.Module and vice versa).  Order within
    # this list only matters relative to containers (DictSerializer etc.)
    # which must come before torch so e.g. a state_dict dispatches to
    # DictSerializer not to a hypothetical dict-of-tensors serializer.
    torch_serializers = [TorchTensorSerializer, TorchModuleSerializer]
    torch_serializers_by_type = {
        "torch.Tensor": TorchTensorSerializer,
        # nn.Module and all subclasses dispatch here via the MRO walk in
        # :class:`TypeDispatchRegistry`.
        "torch.nn.modules.module.Module": TorchModuleSerializer,
    }
