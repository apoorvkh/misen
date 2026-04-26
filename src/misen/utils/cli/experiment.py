"""Helpers for generating experiment CLIs with ``tyro``."""

from __future__ import annotations

import contextlib
import functools
import importlib
import importlib.util
import sys
from dataclasses import dataclass, field, fields, make_dataclass
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast, get_args

import cloudpickle
import tyro
from rich.console import Console
from rich.markup import escape
from rich.pretty import Pretty
from rich.text import Text
from rich.tree import Tree

from misen.exceptions import CacheError
from misen.executor import ExecutorType  # noqa: TC001
from misen.utils.runtime_events import task_label
from misen.utils.settings import Settings
from misen.workspace import WorkspaceType  # noqa: TC001

from . import tui
from .display import format_task_line_markup, iter_task_arg_children

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from misen import Experiment
    from misen.tasks import Task


__all__ = [
    "experiment",
    "experiment_cli",
    "resolve_experiment_class",
    "resolve_experiment_reference",
]


ExperimentCommand = Literal["run", "list", "tree", "count", "result", "incomplete", "logs"]
TreeDepthArg = Annotated[
    int | None,
    tyro.conf.arg(
        name="max-depth",
        aliases=("-L",),
        help="Maximum tree depth. 0 renders only named experiment tasks.",
    ),
]
TreeAllArg = Annotated[
    bool,
    tyro.conf.arg(
        name="all",
        help="Expand shared dependencies under each parent instead of collapsing duplicates.",
    ),
]
TreeCacheableOnlyArg = Annotated[
    bool,
    tyro.conf.arg(
        name="cacheable-only",
        help="Show only cacheable tasks in the tree.",
    ),
]
TreeIncompleteArg = Annotated[
    bool,
    tyro.conf.arg(
        name="incomplete",
        help="Show only incomplete tasks in the tree.",
    ),
]
LogsJobArg = Annotated[
    bool,
    tyro.conf.arg(
        name="job",
        help="Show the task's job log instead of the task log.",
    ),
]
LogsListArg = Annotated[
    bool,
    tyro.conf.arg(
        name="list",
        help="List available log entries with metadata instead of printing content.",
    ),
]
LogsAllArg = Annotated[
    bool,
    tyro.conf.arg(
        name="all",
        help="Show all job logs (only meaningful with --job).",
    ),
]
ListCacheableOnlyArg = Annotated[
    bool,
    tyro.conf.arg(
        name="cacheable-only",
        help="Show only cacheable tasks in the list.",
    ),
]
ListIncompleteArg = Annotated[
    bool,
    tyro.conf.arg(
        name="incomplete",
        help="Show only incomplete tasks in the list.",
    ),
]
RunTuiArg = Annotated[
    bool,
    tyro.conf.arg(
        name="tui",
        help="Enable run TUI; disable with --no-tui to block until jobs terminate.",
    ),
]


@dataclass(frozen=True)
class _ExperimentEntryArgs:
    reference: Annotated[
        str,
        tyro.conf.Positional,
        tyro.conf.arg(
            name="experiment-ref",
            help="Experiment class reference in '<module>:<ExperimentClass>' or '<path.py>:<ExperimentClass>' format.",
        ),
    ]


@dataclass(frozen=True)
class _BootstrapArgs:
    config: Path | None = None
    executor: ExecutorType | str = "auto"
    workspace: WorkspaceType | str = "auto"


@dataclass(frozen=True)
class _ConfigGroup:
    """Holds ``--config`` in its own help panel."""

    config: Path | None = None


@dataclass(frozen=True)
class _WorkspaceGroup:
    """Holds the ``--workspace`` selector in its own help panel."""

    workspace: WorkspaceType | str = "auto"


@dataclass(frozen=True)
class _ExecutorGroup:
    """Holds the ``--executor`` selector in its own help panel."""

    executor: ExecutorType | str = "auto"


@dataclass(frozen=True)
class RunCommandArgs:
    task: tyro.conf.Positional[str | None] = None
    tui: RunTuiArg = True


@dataclass(frozen=True)
class ListCommandArgs:
    cacheable_only: ListCacheableOnlyArg = False
    incomplete: ListIncompleteArg = False


@dataclass(frozen=True)
class TreeCommandArgs:
    task: tyro.conf.Positional[str | None] = None
    all: TreeAllArg = False
    max_depth: TreeDepthArg = None
    cacheable_only: TreeCacheableOnlyArg = False
    incomplete: TreeIncompleteArg = False


@dataclass(frozen=True)
class CountCommandArgs:
    pass


@dataclass(frozen=True)
class ResultCommandArgs:
    task: tyro.conf.Positional[str | None] = None


@dataclass(frozen=True)
class LogsCommandArgs:
    task: tyro.conf.Positional[str | None] = None
    job_id: tyro.conf.Positional[str | None] = None
    job: LogsJobArg = False
    list: LogsListArg = False
    all: LogsAllArg = False


@dataclass(frozen=True)
class IncompleteCommandArgs:
    all: TreeAllArg = False
    max_depth: TreeDepthArg = None
    cacheable_only: TreeCacheableOnlyArg = False


ExperimentCommandArgs = tyro.conf.OmitSubcommandPrefixes[
    Annotated[RunCommandArgs, tyro.conf.subcommand(name="run", prefix_name=False)]
    | Annotated[ListCommandArgs, tyro.conf.subcommand(name="list", prefix_name=False)]
    | Annotated[TreeCommandArgs, tyro.conf.subcommand(name="tree", prefix_name=False)]
    | Annotated[CountCommandArgs, tyro.conf.subcommand(name="count", prefix_name=False)]
    | Annotated[ResultCommandArgs, tyro.conf.subcommand(name="result", prefix_name=False)]
    | Annotated[IncompleteCommandArgs, tyro.conf.subcommand(name="incomplete", prefix_name=False)]
    | Annotated[LogsCommandArgs, tyro.conf.subcommand(name="logs", prefix_name=False)]
]


def _parse_experiment_reference(reference: str) -> tuple[str, str]:
    """Split and validate ``<module-or-file>:<ExperimentClass>`` reference text."""
    module_name, separator, class_name = reference.rpartition(":")
    if not separator or not module_name or not class_name:
        msg = "Invalid experiment reference. Expected '<module-or-file>:<ExperimentClass>'."
        raise ValueError(msg)
    return module_name, class_name


@contextlib.contextmanager
def _prefer_local_src_path() -> Iterator[None]:
    """Temporarily prioritize ``./src`` on ``sys.path`` when it exists."""
    src_dir = (Path.cwd() / "src").resolve()
    if not src_dir.is_dir():
        yield
        return

    src_path = str(src_dir)
    original_index = next((index for index, value in enumerate(sys.path) if value == src_path), None)
    if original_index is not None:
        sys.path.pop(original_index)
    sys.path.insert(0, src_path)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(src_path)
        if original_index is not None:
            sys.path.insert(min(original_index, len(sys.path)), src_path)


def _register_module_pickle_by_value(module: object) -> None:
    """Register module for value pickling when supported by cloudpickle."""
    with contextlib.suppress(ValueError):
        # prefer value-based pickling of Experiment classes
        # so runtime behavior of `misen experiment` is more similar to `python -c "Experiment.cli()"`
        cloudpickle.register_pickle_by_value(module)


def _is_python_file_reference(reference: str) -> bool:
    """Return whether a reference token points to a Python source file."""
    return reference.endswith(".py") or "/" in reference or "\\" in reference


def _resolve_reference_path(reference: str) -> Path:
    """Resolve a file reference into an absolute Python file path."""
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file() or path.suffix != ".py":
        msg = f"Experiment module file {reference!r} does not exist or is not a .py file."
        raise ValueError(msg)
    return path


def _module_name_from_local_src(file_path: Path) -> str | None:
    """Infer importable module path for files inside ``./src``."""
    src_dir = (Path.cwd() / "src").resolve()
    if not src_dir.is_dir() or not file_path.is_relative_to(src_dir):
        return None
    rel_path = file_path.relative_to(src_dir)
    if rel_path.name == "__init__.py":
        return ".".join(rel_path.parent.parts) if rel_path.parent.parts else None
    return ".".join(rel_path.with_suffix("").parts)


def _import_module_from_file(file_path: Path) -> ModuleType:
    """Import a Python module from an explicit file path."""
    module_hash = sha1(str(file_path).encode("utf-8"), usedforsecurity=False).hexdigest()
    module_name = f"_misen_experiment_{module_hash}"
    existing = sys.modules.get(module_name)
    if isinstance(existing, ModuleType):
        return existing

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load module from {str(file_path)!r}."
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _resolve_experiment_module(module_reference: str) -> ModuleType:
    """Resolve module by module path or explicit Python file reference."""
    if not _is_python_file_reference(module_reference):
        with _prefer_local_src_path():
            return importlib.import_module(module_reference)

    file_path = _resolve_reference_path(module_reference)
    local_module_name = _module_name_from_local_src(file_path)
    if local_module_name is not None:
        with _prefer_local_src_path():
            return importlib.import_module(local_module_name)
    return _import_module_from_file(file_path)


def resolve_experiment_reference(reference: str) -> type[Experiment] | Experiment:
    """Resolve ``module:Symbol`` or ``path.py:Symbol`` to an Experiment class or instance.

    Instance references (for example ``my_project.configs.training:__config__``)
    let the shared ``misen experiment`` CLI use the instance's bound field values
    as defaults, matching the ``python -m`` entry point.
    """
    module_reference, attr_name = _parse_experiment_reference(reference)
    module = _resolve_experiment_module(module_reference)
    _register_module_pickle_by_value(module)

    if not hasattr(module, attr_name):
        msg = f"Module {module_reference!r} has no attribute {attr_name!r}."
        raise ValueError(msg)

    from misen import Experiment

    target = getattr(module, attr_name)
    if isinstance(target, type):
        if not issubclass(target, Experiment):
            msg = f"Referenced class {reference!r} is not a misen.Experiment subclass."
            raise TypeError(msg)
        return target
    if isinstance(target, Experiment):
        return target
    msg = f"Referenced symbol {reference!r} is not a misen.Experiment class or instance."
    raise TypeError(msg)


def resolve_experiment_class(reference: str) -> type[Experiment]:
    """Resolve an experiment class from ``module:Class`` or ``path.py:Class`` reference text."""
    target = resolve_experiment_reference(reference)
    if not isinstance(target, type):
        msg = f"Referenced symbol {reference!r} is not a class."
        raise TypeError(msg)
    return cast("type[Experiment]", target)


def _system_exit_code(exc: SystemExit) -> int:
    """Normalize ``SystemExit.code`` into a stable integer exit code."""
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    try:
        return int(code)
    except (TypeError, ValueError):
        return 1


def experiment(argv: list[str] | tuple[str, ...] | None = None) -> int:
    """Run an experiment CLI by resolving ``<module:ExperimentClass>``."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list or args_list[0] in {"-h", "--help"}:
        try:
            tyro.cli(_ExperimentEntryArgs, args=["--help"])
        except SystemExit as exc:
            return _system_exit_code(exc)
        return 0

    parsed, unknown_args = tyro.cli(
        _ExperimentEntryArgs,
        args=args_list,
        return_unknown_args=True,
        add_help=False,
    )

    try:
        experiment_ref = resolve_experiment_reference(parsed.reference)
    except (ImportError, TypeError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    experiment_cli(experiment_ref, argv=unknown_args)
    return 0


def _resolve_command_task(*, command: str, task_name: str | None) -> str:
    """Resolve task selector for command-specific task operations."""
    if task_name is None:
        msg = f"{command!r} command requires a task name."
        raise ValueError(msg)
    return task_name


def _task_sort_key(task: Task[Any]) -> tuple[str, str]:
    """Return stable sort key for task display."""
    return (task.meta.id, task.task_hash().b32())


def _iter_task_closure(root_tasks: Iterable[Task[Any]]) -> list[Task[Any]]:
    """Return all unique tasks reachable from the given roots."""
    visited: set[Task[Any]] = set()

    def visit(task: Task[Any]) -> None:
        if task in visited:
            return
        visited.add(task)
        for dependency in task.dependencies:
            visit(dependency)

    for root in root_tasks:
        visit(root)

    return sorted(visited, key=_task_sort_key)


def _task_done(task: Task[Any], workspace: Any) -> bool:
    """Return completion status for one task in the given workspace."""
    return task.done(workspace=workspace)


def _status_indicator(*, done: bool) -> str:
    """Return rich status marker for one task."""
    return "[green]✓[/green]" if done else "[yellow]○[/yellow]"


def _styled_title(title: str) -> str:
    """Return consistently-styled title text for list/tree outputs."""
    color = "yellow" if "Incomplete" in title else "cyan"
    return f"[bold {color}]{escape(title)}[/bold {color}]"


def _task_display_label(task: Task[Any]) -> str:
    """Return one task label for CLI list/tree output (no hash, no job id)."""
    return format_task_line_markup(task)


def _build_task_tree(
    named_tasks: Mapping[str, Task[Any]],
    workspace: Any,
    *,
    title: str = "Tasks",
    show_all: bool = False,
    max_depth: int | None = None,
    cacheable_only: bool = False,
    incomplete_only: bool = False,
) -> Tree:
    """Build a rich tree for experiment tasks and dependencies."""
    if max_depth is not None and max_depth < 0:
        msg = "Tree max depth must be >= 0."
        raise ValueError(msg)

    tree = Tree(_styled_title(title))
    rendered: set[Task[Any]] = set()

    def include_task(task: Task[Any]) -> bool:
        if cacheable_only and not task.meta.cache:
            return False
        return not (incomplete_only and _task_done(task, workspace))

    def add_task(
        branch: Tree,
        task: Task[Any],
        *,
        arg_prefix: str | None,
        ancestry: set[Task[Any]],
        depth: int,
    ) -> None:
        if not include_task(task):
            return
        if not show_all and task in rendered:
            return

        status = _status_indicator(done=_task_done(task, workspace))
        task_line = format_task_line_markup(task, prefix=arg_prefix)
        line = f"{status} {task_line}"

        if task in ancestry:
            branch.add(f"{line} [red](cycle)[/red]")
            return

        child_branch = branch.add(line)

        if max_depth is not None and depth >= max_depth:
            return

        rendered.add(task)
        child_ancestry = ancestry | {task}
        for child_label, dependency in iter_task_arg_children(task):
            add_task(
                child_branch,
                dependency,
                arg_prefix=child_label,
                ancestry=child_ancestry,
                depth=depth + 1,
            )

    root_tasks = _filter_root_named_tasks(named_tasks)
    rendered_any = False
    for _, task in sorted(root_tasks.items()):
        if not include_task(task):
            continue
        add_task(tree, task, arg_prefix=None, ancestry=set(), depth=0)
        rendered_any = True

    if not rendered_any:
        tree.add("[dim]No tasks matched filters.[/dim]")
    return tree


def _filter_root_named_tasks(named_tasks: Mapping[str, Task[Any]]) -> Mapping[str, Task[Any]]:
    """Return the named tasks that are not dependencies of any other named task.

    A named task is a "root" for tree display when nothing else in the named
    task map transitively depends on it. Intermediate named tasks are still
    reachable through their parents' subtrees.
    """
    non_roots: set[Task[Any]] = set()
    for task in named_tasks.values():
        for dep in _iter_task_closure([task]):
            if dep is not task:
                non_roots.add(dep)
    roots = {name: task for name, task in named_tasks.items() if task not in non_roots}
    return roots or named_tasks


def _build_task_list_lines(
    named_tasks: Mapping[str, Task[Any]],
    workspace: Any,
    *,
    cacheable_only: bool = False,
    incomplete_only: bool = False,
) -> list[str]:
    """Build line-by-line text rows for distinct experiment tasks."""
    lines: list[str] = []

    for task in _iter_task_closure(named_tasks.values()):
        if cacheable_only and not task.meta.cache:
            continue

        done = _task_done(task, workspace)
        if incomplete_only and done:
            continue
        lines.append(f"{_status_indicator(done=done)} {_task_display_label(task)}")

    return lines


def _count_completion(named_tasks: Mapping[str, Task[Any]], workspace: Any) -> tuple[int, int]:
    """Return completed/total counts over distinct tasks in the dependency closure."""
    tasks = _iter_task_closure(named_tasks.values())
    total_count = len(tasks)
    complete_count = sum(1 for task in tasks if _task_done(task, workspace))
    return complete_count, total_count


def _unwrap(value: Any, attr: str) -> Any:
    """Unwrap a single-field help-group dataclass by pulling ``attr`` out.

    Real CLI parsing wraps ``--config``/``--workspace``/``--executor`` in
    ``_ConfigGroup``/``_WorkspaceGroup``/``_ExecutorGroup``; tests pass the raw
    value directly on ``args``. Accept either.
    """
    return getattr(value, attr) if hasattr(value, attr) and not isinstance(value, (str, Path)) else value


def _args_config(args: Any) -> Path | None:
    return _unwrap(args.config, "config")


def _args_workspace(args: Any) -> Any:
    return _unwrap(args.workspace, "workspace")


def _args_executor(args: Any) -> Any:
    return _unwrap(args.executor, "executor")


def _resolve_workspace(args: Any) -> Any:
    """Resolve workspace instance from parsed CLI args."""
    from misen.workspace import Workspace

    settings = Settings(config_file=_args_config(args))
    workspace = _args_workspace(args)
    return Workspace.auto(settings=settings) if workspace == "auto" else workspace


def _resolve_executor(args: Any) -> Any:
    """Resolve executor instance from parsed CLI args."""
    from misen.executor import Executor

    settings = Settings(config_file=_args_config(args))
    executor = _args_executor(args)
    return Executor.auto(settings=settings) if executor == "auto" else executor


_COMMAND_TYPES: dict[ExperimentCommand, type[Any]] = {
    "run": RunCommandArgs,
    "list": ListCommandArgs,
    "tree": TreeCommandArgs,
    "count": CountCommandArgs,
    "result": ResultCommandArgs,
    "incomplete": IncompleteCommandArgs,
    "logs": LogsCommandArgs,
}
_COMMAND_ATTR_PREFIX: dict[str, str] = {"incomplete": "tree"}


def _coerce_command(args: Any) -> Any:
    """Return ``args.command`` as a typed ``*CommandArgs`` dataclass.

    Tests pass ``args.command`` as a string plus flat ``<prefix>_<field>``
    attributes on ``args`` (e.g. ``tree_all``, ``list_incomplete``). This
    helper folds both that test shape and the real parsed-tyro-dataclass
    shape into a single typed value downstream code can consume directly.
    """
    command = args.command
    if isinstance(command, tuple(_COMMAND_TYPES.values())):
        return command
    if not isinstance(command, str):
        msg = f"Unsupported command payload type: {type(command)!r}"
        raise TypeError(msg)

    cls = _COMMAND_TYPES.get(cast("ExperimentCommand", command))
    if cls is None:
        msg = f"Unknown command {command!r}. Expected one of: {sorted(_COMMAND_TYPES)}."
        raise ValueError(msg)
    prefix = _COMMAND_ATTR_PREFIX.get(command, command)
    kwargs = {f.name: getattr(args, f"{prefix}_{f.name}") for f in fields(cls) if hasattr(args, f"{prefix}_{f.name}")}
    return cls(**kwargs)


def _resolve_task_or_hash(experiment: Any, query: str) -> Task[Any]:
    """Resolve a task by exact name or hash prefix.

    Args:
        experiment: Experiment instance.
        query: Task name key or hash prefix.

    Returns:
        Matching task.

    Raises:
        ValueError: If zero or multiple tasks match.
    """
    # Try exact name match first.
    named_tasks = experiment.normalized_tasks()
    if query in named_tasks:
        return named_tasks[query]

    # Fallback: hash prefix match across entire task closure.
    query_upper = query.upper()
    all_tasks = _iter_task_closure(named_tasks.values())
    matches = [t for t in all_tasks if t.task_hash().b32().startswith(query_upper)]

    if len(matches) == 1:
        return matches[0]
    if not matches:
        msg = f"No task matching {query!r}."
        raise ValueError(msg)
    labels = ", ".join(task_label(t) for t in matches)
    msg = f"Ambiguous query {query!r}: matches {labels}."
    raise ValueError(msg)


def _find_work_unit_for_task(experiment: Any, task: Task[Any]) -> Any:
    """Find the work unit containing a given task.

    Args:
        experiment: Experiment instance.
        task: Task to locate.

    Returns:
        WorkUnit containing the task.

    Raises:
        ValueError: If the task is not found in any work unit.
    """
    from misen.utils.work_unit import build_work_graph

    work_graph = build_work_graph(set(experiment.normalized_tasks().values()))
    for work_unit in work_graph.nodes():
        if task == work_unit.root or task in set(work_unit.graph.nodes()):
            return work_unit

    msg = f"No job found containing task {task_label(task)}."
    raise ValueError(msg)


def _list_task_logs(task: Task[Any], workspace: Any, console: Console) -> None:
    """List available task log entries with metadata."""
    try:
        resolved_hash = task.resolved_hash(workspace=workspace).b32()
    except CacheError:
        console.print("  [dim]No resolved hash (task not yet executed).[/dim]")
        return

    log_dir = Path(workspace.directory) / "task_logs" / resolved_hash[:2]
    if not log_dir.exists():
        console.print("  [dim]No logs found.[/dim]")
        return

    log_files = sorted(log_dir.glob(f"{resolved_hash}_*.log"), key=lambda p: p.stat().st_mtime)
    if not log_files:
        console.print("  [dim]No logs found.[/dim]")
        return

    for log_path in log_files:
        _, job_id = log_path.stem.rsplit("_", 1)
        stat = log_path.stat()
        time_str = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(timespec="seconds")
        console.print(
            f"  job_id=[cyan]{escape(job_id)}[/cyan]  modified=[dim]{time_str}[/dim]  size=[dim]{stat.st_size}B[/dim]"
        )


def _list_job_logs(log_paths: list[Path], console: Console) -> None:
    """List available job log entries with metadata."""
    for log_path in log_paths:
        stat = log_path.stat()
        size = stat.st_size
        mtime = stat.st_mtime
        time_str = datetime.fromtimestamp(mtime, tz=UTC).isoformat(timespec="seconds")
        console.print(
            f"  [bright_white]{escape(log_path.name)}[/bright_white]"
            f"  modified=[dim]{time_str}[/dim]"
            f"  size=[dim]{size}B[/dim]"
        )


def _print_log_content(log_path: Path, console: Console, *, rule_title: str | None = None) -> None:
    """Print log file content with optional rule header."""
    if rule_title is not None:
        console.rule(rule_title)
    console.print(Text.from_ansi(log_path.read_text(encoding="utf-8", errors="replace")))


def _execute_command(*, args: Any, console: Console) -> None:
    """Execute resolved experiment CLI command."""
    command = _coerce_command(args)
    match command:
        case RunCommandArgs():
            _cmd_run(command, args)
        case CountCommandArgs():
            _cmd_count(args, console)
        case TreeCommandArgs() | IncompleteCommandArgs():
            _cmd_tree(command, args, console)
        case ListCommandArgs():
            _cmd_list(command, args, console)
        case ResultCommandArgs():
            _cmd_result(command, args, console)
        case LogsCommandArgs():
            _cmd_logs(command, args, console)
        case _:
            msg = f"Unsupported command type {type(command)!r}"
            raise TypeError(msg)


def _cmd_run(command: RunCommandArgs, args: Any) -> None:
    executor = _resolve_executor(args)
    workspace = _resolve_workspace(args)
    if command.task is None:
        if command.tui:
            tui.submit_and_watch_jobs(experiment=args.experiment, executor=executor, workspace=workspace)
        else:
            tui.run_without_tui(experiment=args.experiment, executor=executor, workspace=workspace)
        return
    args.experiment[command.task].submit(executor=executor, workspace=workspace, blocking=True)


def _cmd_count(args: Any, console: Console) -> None:
    workspace = _resolve_workspace(args)
    complete, total = _count_completion(args.experiment.normalized_tasks(), workspace)
    word = "task" if total == 1 else "tasks"
    console.print(f"Completed {complete} of {total} {word}.")


def _cmd_tree(command: TreeCommandArgs | IncompleteCommandArgs, args: Any, console: Console) -> None:
    workspace = _resolve_workspace(args)
    all_named = args.experiment.normalized_tasks()

    tree_task = command.task if isinstance(command, TreeCommandArgs) else None
    if tree_task is not None:
        if tree_task not in all_named:
            msg = f"Experiment has no task named {tree_task!r}. Known tasks: {sorted(all_named)}"
            raise ValueError(msg)
        named_tasks: Mapping[str, Task[Any]] = {tree_task: all_named[tree_task]}
    else:
        named_tasks = all_named

    incomplete_only = isinstance(command, IncompleteCommandArgs) or command.incomplete
    complete, total = _count_completion(named_tasks, workspace)
    if incomplete_only and complete == total:
        console.print("[green]No incomplete tasks.[/green]")
        return

    tree = _build_task_tree(
        named_tasks,
        workspace,
        title="Incomplete Tasks" if incomplete_only else "Tasks",
        show_all=command.all,
        max_depth=command.max_depth,
        cacheable_only=command.cacheable_only,
        incomplete_only=incomplete_only,
    )
    console.print(tree)


def _cmd_list(command: ListCommandArgs, args: Any, console: Console) -> None:
    workspace = _resolve_workspace(args)
    named_tasks = args.experiment.normalized_tasks()
    complete, total = _count_completion(named_tasks, workspace)
    if command.incomplete and complete == total:
        console.print("[green]No incomplete tasks.[/green]")
        return

    title = "Incomplete Tasks" if command.incomplete else "Tasks"
    lines = _build_task_list_lines(
        named_tasks,
        workspace,
        cacheable_only=command.cacheable_only,
        incomplete_only=command.incomplete,
    )
    if not lines:
        console.print("[yellow]No tasks matched filters.[/yellow]")
        return

    console.print(_styled_title(title))
    for line in lines:
        console.print(line)


def _cmd_result(command: ResultCommandArgs, args: Any, console: Console) -> None:
    workspace = _resolve_workspace(args)
    task_key = _resolve_command_task(command="result", task_name=command.task)
    console.print(Pretty(args.experiment.result(task_key, workspace=workspace)))


def _cmd_logs(command: LogsCommandArgs, args: Any, console: Console) -> None:
    workspace = _resolve_workspace(args)
    if command.job:
        _cmd_logs_job_mode(command, args, workspace, console)
    else:
        _cmd_logs_task_mode(command, args, workspace, console)


def _cmd_logs_task_mode(command: LogsCommandArgs, args: Any, workspace: Any, console: Console) -> None:
    if command.task is not None:
        task = _resolve_task_or_hash(args.experiment, command.task)
        _show_task_log(task, workspace, console, job_id=command.job_id, list_mode=command.list)
        return

    printed_any = False
    for task in _iter_task_closure(args.experiment.normalized_tasks().values()):
        if _show_task_log(task, workspace, console, job_id=command.job_id, list_mode=command.list, skip_missing=True):
            printed_any = True
    if not printed_any:
        console.print("[dim]No logs found.[/dim]")


def _show_task_log(
    task: Task[Any],
    workspace: Any,
    console: Console,
    *,
    job_id: str | None,
    list_mode: bool,
    skip_missing: bool = False,
) -> bool:
    """Print one task's log (or its log entry listing). Return whether anything was printed."""
    status = _status_indicator(done=_task_done(task, workspace))
    header = f"{status} {_task_display_label(task)}"
    if list_mode:
        console.print(header)
        _list_task_logs(task, workspace, console)
        return True
    try:
        with workspace.read_task_log(task, job_id=job_id) as f:
            log_content = f.read()
    except FileNotFoundError:
        if skip_missing:
            return False
        console.print("[dim]No logs found.[/dim]")
        return False
    console.rule(header)
    console.print(Text.from_ansi(log_content))
    return True


def _cmd_logs_job_mode(command: LogsCommandArgs, args: Any, workspace: Any, console: Console) -> None:
    if command.task is not None:
        task = _resolve_task_or_hash(args.experiment, command.task)
        work_unit = _find_work_unit_for_task(args.experiment, task)
        log_paths = sorted(workspace.job_log_iter(work_unit), key=lambda p: p.stat().st_mtime)
    else:
        log_paths = sorted(workspace.job_log_iter(), key=lambda p: p.stat().st_mtime)

    if command.job_id is not None:
        log_paths = [p for p in log_paths if f"_{command.job_id}.log" in p.name]

    if not log_paths:
        console.print("[dim]No job logs found.[/dim]")
        return

    if command.list:
        _list_job_logs(log_paths, console)
        return

    for log_path in log_paths if command.all else log_paths[-1:]:
        _print_log_content(log_path, console, rule_title=f"[bright_white]{escape(log_path.name)}[/bright_white]")


def _resolve_command(*, command_token: str | None, unknown_args: list[str]) -> ExperimentCommand:
    """Resolve command from first parse token and unknown argv tail."""
    available_commands = set(get_args(ExperimentCommand))

    if command_token in available_commands:
        return cast("ExperimentCommand", command_token)

    if command_token not in {None, "[]"}:
        msg = f"Unknown command {command_token!r}. Expected one of: {sorted(available_commands)}."
        raise ValueError(msg)

    for token in unknown_args:
        if token in available_commands:
            return cast("ExperimentCommand", token)

    return "run"


def experiment_cli(
    experiment_ref: type[Experiment] | Experiment,
    argv: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Parse CLI args and execute experiment command.

    Args:
        experiment_ref: Experiment class, or an instance whose bound field
            values should seed the CLI defaults (each field still overridable).
        argv: Optional command tokens; defaults to ``sys.argv[1:]`` when ``None``.

    Notes:
        Parsing happens in two phases so executor/workspace concrete argument
        schemas can be selected dynamically from ``--executor``/``--workspace`` flags.
    """
    from misen.executor import Executor
    from misen.workspace import Workspace

    if isinstance(experiment_ref, type):
        experiment_cls: type[Experiment] = cast("type[Experiment]", experiment_ref)
        experiment_default: Experiment | None = None
    else:
        experiment_cls = type(experiment_ref)
        experiment_default = experiment_ref

    args_list = list(sys.argv[1:] if argv is None else argv)
    bootstrap_args, unknown_args = tyro.cli(
        _BootstrapArgs,
        add_help=False,
        return_unknown_args=True,
        args=args_list,
    )
    command_default_factories: dict[ExperimentCommand, type[object]] = {
        "run": RunCommandArgs,
        "list": ListCommandArgs,
        "tree": TreeCommandArgs,
        "count": CountCommandArgs,
        "result": ResultCommandArgs,
        "incomplete": IncompleteCommandArgs,
        "logs": LogsCommandArgs,
    }
    resolved_command = _resolve_command(command_token=None, unknown_args=unknown_args)

    # Each CLI group lives in its own single-field wrapper so tyro renders a
    # dedicated help panel ("config options", "workspace options", ...). Field
    # order drives help-panel order. ``kw_only=True`` on the generated class
    # lets us mix required (concrete executor/workspace) and defaulted fields
    # without reordering.
    cli_fields: list[tuple[Any, ...]] = [
        (
            "config",
            tyro.conf.OmitArgPrefixes[_ConfigGroup],
            field(default_factory=lambda: _ConfigGroup(config=bootstrap_args.config)),
        ),
    ]
    if bootstrap_args.workspace == "auto":
        cli_fields.append(
            (
                "workspace",
                tyro.conf.OmitArgPrefixes[_WorkspaceGroup],
                field(default_factory=lambda: _WorkspaceGroup(workspace=bootstrap_args.workspace)),
            )
        )
    else:
        cli_fields.append(("workspace", Workspace.resolve_type(bootstrap_args.workspace)))
    if bootstrap_args.executor == "auto":
        cli_fields.append(
            (
                "executor",
                tyro.conf.OmitArgPrefixes[_ExecutorGroup],
                field(default_factory=lambda: _ExecutorGroup(executor=bootstrap_args.executor)),
            )
        )
    else:
        cli_fields.append(("executor", Executor.resolve_type(bootstrap_args.executor)))
    experiment_spec = field(default=experiment_default) if experiment_default is not None else field()
    cli_fields.extend(
        [
            ("experiment", tyro.conf.OmitArgPrefixes[experiment_cls], experiment_spec),  # ty:ignore[invalid-type-form]
            ("command", ExperimentCommandArgs, field(default_factory=command_default_factories[resolved_command])),
        ]
    )

    console = Console()
    parsed = tyro.cli(make_dataclass("", cli_fields, kw_only=True), args=unknown_args)
    _execute_command(args=cast("Any", parsed), console=console)


class _ClassOrInstanceMethod:
    """Descriptor: passes the class on class access and the instance on instance access.

    Lets a single method serve both ``Experiment.cli()`` (class → field defaults)
    and ``config.cli()`` (instance → bound field values seed CLI defaults).
    """

    def __init__(self, func: Callable[[Any], None]) -> None:
        self._func = func

    def __get__(self, instance: Any, owner: type | None = None) -> Callable[[], None]:
        bound = functools.partial(self._func, instance if instance is not None else owner)
        functools.update_wrapper(bound, self._func)
        return bound
