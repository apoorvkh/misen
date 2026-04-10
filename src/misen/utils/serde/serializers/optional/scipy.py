"""Serializer for scipy sparse matrices."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["scipy_serializers", "scipy_serializers_by_type"]

scipy_serializers: SerializerTypeList = []
scipy_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("scipy") is not None:
    from pathlib import Path

    class ScipySparseSerializer(Serializer[Any]):
        """Serialize scipy sparse matrices via ``save_npz``/``load_npz``."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import scipy.sparse

            return scipy.sparse.issparse(obj)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import scipy
            import scipy.sparse

            scipy.sparse.save_npz(directory / "data.npz", obj)
            write_meta(directory, ScipySparseSerializer, scipy_version=scipy.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            import scipy.sparse

            return scipy.sparse.load_npz(directory / "data.npz")

    scipy_serializers = [ScipySparseSerializer]
    scipy_serializers_by_type = {}  # scipy sparse types vary, rely on match()
