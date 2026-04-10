"""Serializer for sympy expressions via srepr."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["sympy_serializers", "sympy_serializers_by_type"]

sympy_serializers: SerializerTypeList = []
sympy_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("sympy") is not None:
    from pathlib import Path

    class SympyExprSerializer(Serializer[Any]):
        """Serialize sympy expressions via ``srepr`` (canonical string representation)."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import sympy

            return isinstance(obj, sympy.Basic)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import sympy

            s = sympy.srepr(obj)
            (directory / "data.sympy").write_text(s, encoding="utf-8")
            write_meta(directory, SympyExprSerializer, sympy_version=sympy.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            import sympy

            s = (directory / "data.sympy").read_text(encoding="utf-8")
            return sympy.parse_expr(s, evaluate=False)

    sympy_serializers = [SympyExprSerializer]
    sympy_serializers_by_type = {"sympy.core.basic.Basic": SympyExprSerializer}
