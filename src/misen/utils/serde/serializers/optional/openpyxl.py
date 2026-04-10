"""Serializer for openpyxl Workbook via .xlsx format."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["openpyxl_serializers", "openpyxl_serializers_by_type"]

openpyxl_serializers: SerializerTypeList = []
openpyxl_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("openpyxl") is not None:
    from pathlib import Path

    class OpenpyxlWorkbookSerializer(Serializer[Any]):
        """Serialize ``openpyxl.Workbook`` via ``.xlsx`` format."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            from openpyxl import Workbook

            return isinstance(obj, Workbook)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import openpyxl

            obj.save(directory / "workbook.xlsx")
            write_meta(directory, OpenpyxlWorkbookSerializer, openpyxl_version=openpyxl.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            from openpyxl import load_workbook

            return load_workbook(directory / "workbook.xlsx")

    openpyxl_serializers = [OpenpyxlWorkbookSerializer]
    openpyxl_serializers_by_type = {"openpyxl.workbook.workbook.Workbook": OpenpyxlWorkbookSerializer}
