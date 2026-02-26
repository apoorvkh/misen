"""Implementation for the ``misen fill`` CLI command."""

from __future__ import annotations

import base64
import fnmatch
import os
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import libcst as cst
import tyro  # noqa: TC002
from libcst import matchers as m
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from typing import Any

EXCLUDED_DIRECTORY_NAMES = {
    ".cache",
    ".codex",
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".uv-cache",
    ".venv",
    ".misen",
    "__pycache__",
    "build",
    "dist",
    "site-packages",
}
DEFAULT_ROOT = Path.cwd()
DEFAULT_MODULE_ROOT = "src"
DEFAULT_SOURCE_EXCLUDES = ["__pycache__", "*.pyc", "*.pyo"]


@dataclass(frozen=True, slots=True)
class FillReport:
    """Summary of a ``misen fill`` execution."""

    scanned_files: int
    changed_files: int
    updated_decorators: int
    failed_files: list[tuple[Path, str]]


class BuildSelectionError(RuntimeError):
    """Raised when build-backed file selection fails."""


@dataclass(frozen=True, slots=True)
class BuildSelectionConfig:
    """Resolved uv_build include/exclude settings from ``pyproject.toml``."""

    project_root: Path
    module_root: Path
    module_names: list[str] | None
    source_include: list[str]
    source_exclude: list[str]


class TaskIdFiller(cst.CSTTransformer):
    """Insert UUID ``id=...`` values into ``@task`` decorators."""

    def __init__(self, uuid_factory: Callable[[], str] | None = None) -> None:
        """Initialize rewrite counters."""
        self._uuid_factory = uuid_factory or _default_uuid_factory
        self.updated_decorators = 0

    @override
    def leave_Decorator(self, original_node: cst.Decorator, updated_node: cst.Decorator) -> cst.Decorator:
        """Rewrite task decorators missing an ``id`` or using ``id=None``."""
        del original_node
        decorator = updated_node.decorator
        if _is_task_reference(decorator):
            self.updated_decorators += 1
            return updated_node.with_changes(
                decorator=cst.Call(func=decorator, args=[_make_id_arg(self._uuid_factory())]),
            )

        if not isinstance(decorator, cst.Call) or not _is_task_reference(decorator.func):
            return updated_node

        args = list(decorator.args)
        for index, arg in enumerate(args):
            if arg.keyword is None or arg.keyword.value != "id":
                continue
            if m.matches(arg.value, m.Name("None")):
                args[index] = arg.with_changes(
                    equal=_id_assign_equal(),
                    value=_make_id_string(self._uuid_factory()),
                )
                self.updated_decorators += 1
                return updated_node.with_changes(decorator=decorator.with_changes(args=args))
            return updated_node

        args = _insert_id_arg(args, _make_id_arg(self._uuid_factory()))
        self.updated_decorators += 1
        return updated_node.with_changes(decorator=decorator.with_changes(args=args))


def fill_task_ids_in_source(source: str, uuid_factory: Callable[[], str] | None = None) -> tuple[str, int]:
    """Rewrite a Python module source string and report changed decorators."""
    module = cst.parse_module(source)
    transformer = TaskIdFiller(uuid_factory=uuid_factory)
    updated_module = module.visit(transformer)
    return updated_module.code, transformer.updated_decorators


def fill(
    paths: tyro.conf.Positional[tuple[Path, ...]] = (DEFAULT_ROOT,),
) -> int:
    """Fill missing ``@task(id=...)`` values in Python files."""
    try:
        report = fill_paths_task_ids(paths=paths)
    except BuildSelectionError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    sys.stdout.write(
        "Updated "
        f"{report.updated_decorators} task decorator(s) across {report.changed_files} file(s) "
        f"(scanned {report.scanned_files} Python file(s)).\n",
    )
    for file_path, error in report.failed_files:
        sys.stderr.write(f"Skipped {file_path}: {error}\n")

    return 1 if report.failed_files else 0


def fill_paths_task_ids(
    paths: Iterable[Path],
    uuid_factory: Callable[[], str] | None = None,
) -> FillReport:
    """Fill task ids across Python files addressed by explicit paths."""
    return _fill_file_paths(_iter_python_files_from_paths(paths), uuid_factory=uuid_factory)


def fill_project_task_ids(
    root: Path,
    uuid_factory: Callable[[], str] | None = None,
) -> FillReport:
    """Fill task ids across Python files selected by uv build-backend settings."""
    return _fill_file_paths(iter_build_python_files(root), uuid_factory=uuid_factory)


def _fill_file_paths(
    file_paths: Iterable[Path],
    uuid_factory: Callable[[], str] | None = None,
) -> FillReport:
    """Rewrite ``@task`` ids in a file stream and return a summary."""
    scanned_files = 0
    changed_files = 0
    updated_decorators = 0
    failed_files: list[tuple[Path, str]] = []

    for file_path in file_paths:
        scanned_files += 1
        try:
            original_source = file_path.read_text(encoding="utf-8")
            updated_source, replacements = fill_task_ids_in_source(original_source, uuid_factory=uuid_factory)
        except (OSError, UnicodeError, cst.ParserSyntaxError) as exc:
            failed_files.append((file_path, str(exc)))
            continue

        if replacements == 0:
            continue

        file_path.write_text(updated_source, encoding="utf-8")
        changed_files += 1
        updated_decorators += replacements

    return FillReport(
        scanned_files=scanned_files,
        changed_files=changed_files,
        updated_decorators=updated_decorators,
        failed_files=failed_files,
    )


def _iter_python_files_from_paths(paths: Iterable[Path]) -> Iterator[Path]:
    seen_files: set[Path] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.exists():
            msg = f"Path does not exist: {path}"
            raise BuildSelectionError(msg)

        if path.is_dir():
            candidate_files = iter_build_python_files(path)
        elif path.is_file():
            candidate_files = [path] if path.suffix == ".py" else []
        else:
            msg = f"Unsupported path type: {path}"
            raise BuildSelectionError(msg)

        for candidate in candidate_files:
            resolved_candidate = candidate.resolve()
            if resolved_candidate in seen_files:
                continue
            seen_files.add(resolved_candidate)
            yield resolved_candidate


def iter_project_python_files(root: Path) -> Iterator[Path]:
    """Yield Python files in ``root`` while pruning common non-project folders."""
    for directory_path, directory_names, file_names in os.walk(root):
        directory_names[:] = sorted(name for name in directory_names if name not in EXCLUDED_DIRECTORY_NAMES)
        for file_name in sorted(file_names):
            if file_name.endswith(".py"):
                yield Path(directory_path) / file_name


def iter_build_python_files(root: Path) -> Iterator[Path]:
    """Yield Python files from uv_build include/exclude settings."""
    config = _load_build_selection_config(root.resolve())
    included_files: set[Path] = set()

    include_patterns = [
        *_module_include_patterns(config),
        *config.source_include,
    ]
    for pattern in include_patterns:
        included_files.update(_iter_python_files_matching_pattern(root=config.project_root, pattern=pattern))

    for file_path in sorted(included_files):
        if not _is_within_root(file_path, config.project_root):
            continue
        if _is_source_excluded(
            relative_path=file_path.relative_to(config.project_root),
            source_exclude_patterns=config.source_exclude,
        ):
            continue
        yield file_path


def _load_build_selection_config(root: Path) -> BuildSelectionConfig:
    pyproject_file = root / "pyproject.toml"
    if not pyproject_file.is_file():
        msg = f"Could not determine build-included files: missing {pyproject_file}."
        raise BuildSelectionError(msg)

    pyproject_data = _load_pyproject_data(pyproject_file)
    if not isinstance(pyproject_data, dict):
        msg = f"Could not parse {pyproject_file} as a TOML table."
        raise BuildSelectionError(msg)

    project_table = _as_dict(pyproject_data.get("project"))
    project_name = project_table.get("name")
    if not isinstance(project_name, str) or not project_name:
        msg = f"Could not determine build-included files: `[project].name` missing in {pyproject_file}."
        raise BuildSelectionError(msg)

    uv_build_table = _as_dict(_as_dict(_as_dict(pyproject_data.get("tool")).get("uv")).get("build-backend"))
    module_root = Path(_as_str(uv_build_table.get("module-root")) or DEFAULT_MODULE_ROOT)
    namespace = bool(uv_build_table.get("namespace", False))
    module_names = _resolve_module_names(
        raw_module_name=uv_build_table.get("module-name"),
        default_module_name=_default_module_name(project_name),
        namespace=namespace,
    )

    source_include = _as_str_list(uv_build_table.get("source-include"))
    source_exclude = _as_str_list(uv_build_table.get("source-exclude"))
    if bool(uv_build_table.get("default-excludes", True)):
        source_exclude = [*source_exclude, *DEFAULT_SOURCE_EXCLUDES]

    return BuildSelectionConfig(
        project_root=root,
        module_root=module_root,
        module_names=module_names,
        source_include=source_include,
        source_exclude=source_exclude,
    )


def _load_pyproject_data(pyproject_file: Path) -> dict[str, Any]:
    import importlib

    try:
        toml_module = importlib.import_module("tomllib")
    except ModuleNotFoundError:
        toml_module = importlib.import_module("tomli")

    try:
        return toml_module.loads(pyproject_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, toml_module.TOMLDecodeError) as exc:
        msg = f"Failed parsing {pyproject_file}: {exc}"
        raise BuildSelectionError(msg) from exc


def _resolve_module_names(
    raw_module_name: object,
    default_module_name: str,
    *,
    namespace: bool,
) -> list[str] | None:
    if raw_module_name is None:
        if namespace:
            return None
        return [default_module_name]
    if isinstance(raw_module_name, str):
        return [raw_module_name]
    if isinstance(raw_module_name, list):
        return [value for value in raw_module_name if isinstance(value, str)]
    return [default_module_name]


def _default_module_name(project_name: str) -> str:
    if project_name.endswith("-stubs"):
        return project_name.removesuffix("-stubs")
    return project_name.replace("-", "_").replace(".", "_")


def _module_include_patterns(config: BuildSelectionConfig) -> list[str]:
    patterns: list[str] = []
    if config.module_names is None:
        patterns.append(f"{config.module_root.as_posix()}/**")
        return patterns

    for module_name in config.module_names:
        module_path = config.module_root / Path(*module_name.split("."))
        module_pattern = module_path.as_posix()
        patterns.append(module_pattern)
        patterns.append(f"{module_pattern}.py")
        patterns.append(f"{module_pattern}/**")
    return patterns


def _iter_python_files_matching_pattern(root: Path, pattern: str) -> Iterator[Path]:
    glob_pattern = pattern.removeprefix("/")
    for matched_path in root.glob(glob_pattern):
        yield from _iter_python_files_under(matched_path)


def _iter_python_files_under(path: Path) -> Iterator[Path]:
    resolved_path = path.resolve()
    if resolved_path.is_file():
        if resolved_path.suffix == ".py":
            yield resolved_path
        return
    if not resolved_path.is_dir():
        return
    for file_path in sorted(resolved_path.rglob("*.py")):
        if file_path.is_file():
            yield file_path


def _is_source_excluded(relative_path: Path, source_exclude_patterns: Iterable[str]) -> bool:
    relative_posix = relative_path.as_posix()
    return any(_matches_exclude_pattern(relative_posix, pattern) for pattern in source_exclude_patterns)


def _matches_exclude_pattern(relative_posix: str, pattern: str) -> bool:
    if pattern.startswith("/"):
        return _matches_anchored(relative_posix, pattern.removeprefix("/"))
    return _matches_unanchored(relative_posix, pattern)


def _matches_anchored(relative_posix: str, pattern: str) -> bool:
    if fnmatch.fnmatch(relative_posix, pattern):
        return True
    return any(fnmatch.fnmatch(prefix, pattern) for prefix in _path_prefixes(relative_posix))


def _matches_unanchored(relative_posix: str, pattern: str) -> bool:
    if fnmatch.fnmatch(relative_posix, pattern):
        return True

    return any(fnmatch.fnmatch(segment, pattern) for segment in _path_subpaths(relative_posix))


def _path_prefixes(relative_posix: str) -> Iterator[str]:
    parts = relative_posix.split("/")
    for index in range(1, len(parts)):
        yield "/".join(parts[:index])


def _path_subpaths(relative_posix: str) -> Iterator[str]:
    parts = relative_posix.split("/")
    for start in range(len(parts)):
        for stop in range(start + 1, len(parts) + 1):
            yield "/".join(parts[start:stop])


def _is_within_root(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _as_dict(value: object) -> dict[str, Any]:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_str_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, str)]
    return []


def _is_task_reference(expression: cst.BaseExpression) -> bool:
    return m.matches(expression, m.Name("task")) or m.matches(expression, m.Attribute(attr=m.Name("task")))


def _insert_id_arg(args: list[cst.Arg], id_arg: cst.Arg) -> list[cst.Arg]:
    insertion_index = 0
    while insertion_index < len(args):
        arg = args[insertion_index]
        if arg.keyword is None and arg.star == "":
            insertion_index += 1
            continue
        break

    if insertion_index < len(args):
        return [*args[:insertion_index], id_arg, *args[insertion_index:]]
    return [*args, id_arg]


def _make_id_arg(value: str) -> cst.Arg:
    return cst.Arg(
        keyword=cst.Name("id"),
        equal=_id_assign_equal(),
        value=_make_id_string(value),
    )


def _make_id_string(value: str) -> cst.SimpleString:
    return cst.SimpleString(f'"{value}"')


def _default_uuid_factory() -> str:
    return str(base64.b32encode(secrets.token_bytes(6)).decode("ascii").rstrip("="))


def _id_assign_equal() -> cst.AssignEqual:
    return cst.AssignEqual(
        whitespace_before=cst.SimpleWhitespace(""),
        whitespace_after=cst.SimpleWhitespace(""),
    )
