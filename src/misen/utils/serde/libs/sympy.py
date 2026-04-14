"""Serializer for sympy expressions via srepr."""

import importlib.util
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import (
    Serializer,
    SerializerTypeRegistry,
)

__all__ = ["sympy_serializers", "sympy_serializers_by_type"]

sympy_serializers: list[type[Serializer]] = []
sympy_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("sympy") is not None:
    from pathlib import Path

    class SympyExprSerializer(Serializer[Any]):
        """Serialize sympy expressions via ``srepr`` (canonical string representation)."""

        @staticmethod
        def match(obj: Any) -> bool:
            import sympy

            return isinstance(obj, sympy.Basic)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import sympy

            s = sympy.srepr(obj)
            (directory / "data.sympy").write_text(s, encoding="utf-8")
            return {"sympy_version": sympy.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import sympy

            s = (directory / "data.sympy").read_text(encoding="utf-8")
            return sympy.parse_expr(s, evaluate=False)

    sympy_serializers = [SympyExprSerializer]
    sympy_serializers_by_type = {"sympy.core.basic.Basic": SympyExprSerializer}
