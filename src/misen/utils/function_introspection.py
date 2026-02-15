"""Function-introspection helpers for stable task identifiers."""

from __future__ import annotations

from ast import Lambda, dump, parse, walk
from contextlib import suppress
from hashlib import sha256
from inspect import getfile, getsourcefile, getsourcelines, unwrap
from pathlib import Path
from site import getusersitepackages
from sysconfig import get_paths
from textwrap import dedent
from types import FunctionType
from typing import TypeGuard

__all__ = [
    "canonical_lambda_ast_representation",
    "external_callable_id",
    "is_function_object",
    "is_lambda_function",
    "is_local_project_function",
    "lambda_task_id",
]


def _external_library_roots() -> tuple[Path, ...]:
    """Return normalized roots for stdlib and installed libraries."""
    paths = get_paths()
    roots = {
        Path(path).resolve()
        for key in ("stdlib", "platstdlib", "purelib", "platlib")
        if (path := paths.get(key)) is not None
    }
    roots.add(Path(getusersitepackages()).resolve())
    return tuple(roots)


_EXTERNAL_LIBRARY_ROOTS = _external_library_roots()


def is_function_object(obj: object) -> TypeGuard[FunctionType]:
    """Return whether object is a Python function (including lambdas)."""
    return isinstance(obj, FunctionType)


def is_local_project_function(func: FunctionType) -> bool:
    """Return whether callable source appears to come from project code.

    Args:
        func: Function object to inspect.

    Returns:
        ``True`` if source path is outside known stdlib/site-packages roots.
    """
    source_callable: object = func
    with suppress(ValueError):
        source_callable = unwrap(func)

    if not is_function_object(source_callable):
        return False

    try:
        source_file = getsourcefile(source_callable) or getfile(source_callable)
        source_path = Path(source_file).resolve()
    except (OSError, TypeError):
        return False
    return not any(source_path.is_relative_to(root) for root in _EXTERNAL_LIBRARY_ROOTS)


def is_lambda_function(func: FunctionType) -> bool:
    """Return whether callable is a lambda function."""
    return func.__name__ == "<lambda>"


def canonical_lambda_ast_representation(func: FunctionType) -> str:
    """Return canonical AST representation for a lambda.

    Args:
        func: Lambda function object.

    Returns:
        Deterministic AST dump string.

    Raises:
        ValueError: If source cannot be retrieved or lambda node is not found.
    """
    try:
        source_lines, start_line = getsourcelines(func)
    except (OSError, TypeError) as exc:
        msg = f"Could not retrieve source for lambda {func.__module__}.{func.__qualname__}."
        raise ValueError(msg) from exc

    source = dedent("".join(source_lines))
    module_ast = parse(source)
    lambda_nodes = [node for node in walk(module_ast) if isinstance(node, Lambda)]

    if not lambda_nodes:
        msg = f"Could not find lambda AST for {func.__module__}.{func.__qualname__}."
        raise ValueError(msg)

    target_line = func.__code__.co_firstlineno
    line_matches = [node for node in lambda_nodes if (start_line + node.lineno - 1) == target_line]
    node = line_matches[0] if line_matches else lambda_nodes[0]
    return dump(node, include_attributes=False)


def lambda_task_id(func: FunctionType) -> str:
    """Return deterministic task id for a lambda function."""
    canonical_ast = canonical_lambda_ast_representation(func)
    digest = sha256(canonical_ast.encode()).hexdigest()
    return f"lambda.{digest}"


def external_callable_id(func: FunctionType) -> str:
    """Return default identifier for an external callable."""
    return f"{func.__module__}.{func.__qualname__}"
