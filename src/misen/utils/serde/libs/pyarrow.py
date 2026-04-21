"""Serializer for PyArrow tables."""

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.utils.serde.base import Serializer

__all__ = ["pyarrow_serializers", "pyarrow_serializers_by_type"]

pyarrow_serializers: list[type[Serializer]] = []
pyarrow_serializers_by_type: dict[str, type[Serializer]] = {}

if importlib.util.find_spec("pyarrow") is not None:

    class PyArrowTableSerializer(Serializer[Any]):
        """Serialize ``pyarrow.Table`` via Parquet."""

        @staticmethod
        def match(obj: Any) -> bool:
            import pyarrow as pa

            return isinstance(obj, pa.Table)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(obj, directory / "data.parquet")
            return {"pyarrow_version": pa.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import pyarrow.parquet as pq

            return pq.read_table(directory / "data.parquet")

    pyarrow_serializers = [PyArrowTableSerializer]
    pyarrow_serializers_by_type = {"pyarrow.lib.Table": PyArrowTableSerializer}
