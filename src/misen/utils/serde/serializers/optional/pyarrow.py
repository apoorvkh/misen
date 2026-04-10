"""Serializer for PyArrow tables."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["pyarrow_serializers", "pyarrow_serializers_by_type"]

pyarrow_serializers: SerializerTypeList = []
pyarrow_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("pyarrow") is not None:
    from pathlib import Path

    class PyArrowTableSerializer(Serializer[Any]):
        """Serialize ``pyarrow.Table`` via Parquet."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import pyarrow as pa

            return isinstance(obj, pa.Table)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(obj, directory / "data.parquet")
            write_meta(directory, PyArrowTableSerializer, pyarrow_version=pa.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            import pyarrow.parquet as pq

            return pq.read_table(directory / "data.parquet")

    pyarrow_serializers = [PyArrowTableSerializer]
    pyarrow_serializers_by_type = {"pyarrow.lib.Table": PyArrowTableSerializer}
