"""Serializer for scipy sparse matrices."""

import importlib.util
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import (
    Serializer,
    SerializerTypeRegistry,
)

__all__ = ["scipy_serializers", "scipy_serializers_by_type"]

scipy_serializers: list[type[Serializer]] = []
scipy_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("scipy") is not None:
    from pathlib import Path

    class ScipySparseSerializer(Serializer[Any]):
        """Serialize scipy sparse matrices via ``save_npz``/``load_npz``."""

        @staticmethod
        def match(obj: Any) -> bool:
            import scipy.sparse

            return scipy.sparse.issparse(obj)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import scipy
            import scipy.sparse

            scipy.sparse.save_npz(directory / "data.npz", obj)
            return {"scipy_version": scipy.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import scipy.sparse

            return scipy.sparse.load_npz(directory / "data.npz")

    scipy_serializers = [ScipySparseSerializer]
    scipy_serializers_by_type = {}  # scipy sparse types vary, rely on match()
