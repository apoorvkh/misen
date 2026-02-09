"""Function-introspection helpers used by task wrapping logic."""

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


def _external_library_roots() -> tuple[Path, ...]:
    """Return normalized roots that typically contain stdlib/installed libraries."""
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
    """Return True if object is a Python function object (including lambdas)."""
    return isinstance(obj, FunctionType)


def is_local_project_function(func: FunctionType) -> bool:
    """Return True if callable source appears to come from user project code, not stdlib/site-packages."""
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
    """Return True if the given callable is a lambda function."""
    return func.__name__ == "<lambda>"


def canonical_lambda_ast_representation(func: FunctionType) -> str:
    """Return a canonical AST string for a lambda function."""
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
    """Return a deterministic task id for a lambda function."""
    canonical_ast = canonical_lambda_ast_representation(func)
    digest = sha256(canonical_ast.encode()).hexdigest()
    return f"lambda.{digest}"


def external_callable_id(func: FunctionType) -> str:
    """Return a default id for an external function."""
    return f"{func.__module__}.{func.__qualname__}"
