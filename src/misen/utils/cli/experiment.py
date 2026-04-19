"""Helpers for generating experiment CLIs with ``tyro``."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass, field, make_dataclass
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
from misen.executor import ExecutorType
from misen.utils.runtime_events import task_label
from misen.utils.settings import Settings
from misen.workspace import WorkspaceType

from . import tui
from .display import format_task_line_markup, iter_task_arg_children

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from misen import Experiment
    from misen.tasks import Task


__all__ = [
    "experiment",
    "experiment_cli",
    "resolve_experiment_class",
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


def resolve_experiment_class(reference: str) -> type[Experiment]:
    """Resolve an experiment class from ``module:Class`` or ``path.py:Class`` reference text."""
    module_reference, class_name = _parse_experiment_reference(reference)
    module = _resolve_experiment_module(module_reference)
    _register_module_pickle_by_value(module)

    if not hasattr(module, class_name):
        msg = f"Module {module_reference!r} has no attribute {class_name!r}."
        raise ValueError(msg)

    experiment_cls = getattr(module, class_name)
    if not isinstance(experiment_cls, type):
        msg = f"Referenced symbol {reference!r} is not a class."
        raise TypeError(msg)

    from misen import Experiment

    if not issubclass(experiment_cls, Experiment):
        msg = f"Referenced class {reference!r} is not a misen.Experiment subclass."
        raise TypeError(msg)

    return experiment_cls


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
        experiment_cls = resolve_experiment_class(parsed.reference)
    except (ImportError, TypeError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    experiment_cli(experiment_cls, argv=unknown_args)
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
    used_shared_marker = False

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


def _format_count_message(*, complete_count: int, total_count: int) -> str:
    """Return human-readable completion summary for ``count`` command."""
    task_word = "task" if total_count == 1 else "tasks"
    return f"Completed {complete_count} of {total_count} {task_word}."


def _resolve_workspace(args: Any) -> Any:
    """Resolve workspace instance from parsed CLI args."""
    from misen.workspace import Workspace

    settings = Settings(config_file=args.config)
    return Workspace.auto(settings=settings) if args.workspace == "auto" else args.workspace


def _resolve_executor(args: Any) -> Any:
    """Resolve executor instance from parsed CLI args."""
    from misen.executor import Executor

    settings = Settings(config_file=args.config)
    return Executor.auto(settings=settings) if args.executor == "auto" else args.executor


def _command_name(command: object) -> ExperimentCommand:
    """Normalize command value into a stable literal command name."""
    if isinstance(command, str):
        available_commands = set(get_args(ExperimentCommand))
        if command in available_commands:
            return cast("ExperimentCommand", command)
        msg = f"Unknown command {command!r}. Expected one of: {sorted(available_commands)}."
        raise ValueError(msg)

    command_type_to_name: dict[type[object], ExperimentCommand] = {
        RunCommandArgs: "run",
        ListCommandArgs: "list",
        TreeCommandArgs: "tree",
        CountCommandArgs: "count",
        ResultCommandArgs: "result",
        IncompleteCommandArgs: "incomplete",
        LogsCommandArgs: "logs",
    }
    command_name = command_type_to_name.get(type(command))
    if command_name is not None:
        return command_name

    msg = f"Unsupported command payload type: {type(command)!r}"
    raise TypeError(msg)


def _run_task(args: Any) -> str | None:
    command = args.command
    if isinstance(command, RunCommandArgs):
        return command.task
    return cast("str | None", getattr(args, "run_task", None))


def _run_tui(args: Any) -> bool:
    command = args.command
    if isinstance(command, RunCommandArgs):
        return command.tui
    return bool(getattr(args, "run_tui", True))


def _tree_all(args: Any) -> bool:
    command = args.command
    if isinstance(command, (TreeCommandArgs, IncompleteCommandArgs)):
        return command.all
    return bool(getattr(args, "tree_all", False))


def _tree_max_depth(args: Any) -> int | None:
    command = args.command
    if isinstance(command, (TreeCommandArgs, IncompleteCommandArgs)):
        return command.max_depth
    return cast("int | None", getattr(args, "tree_max_depth", None))


def _tree_cacheable_only(args: Any) -> bool:
    command = args.command
    if isinstance(command, (TreeCommandArgs, IncompleteCommandArgs)):
        return command.cacheable_only
    return bool(getattr(args, "tree_cacheable_only", False))


def _tree_incomplete(args: Any) -> bool:
    command = args.command
    if isinstance(command, TreeCommandArgs):
        return command.incomplete
    if isinstance(command, IncompleteCommandArgs):
        return True
    return bool(getattr(args, "tree_incomplete", False))


def _tree_task(args: Any) -> str | None:
    command = args.command
    if isinstance(command, TreeCommandArgs):
        return command.task
    return cast("str | None", getattr(args, "tree_task", None))


def _list_cacheable_only(args: Any) -> bool:
    command = args.command
    if isinstance(command, ListCommandArgs):
        return command.cacheable_only
    return bool(getattr(args, "list_cacheable_only", False))


def _list_incomplete(args: Any) -> bool:
    command = args.command
    if isinstance(command, ListCommandArgs):
        return command.incomplete
    return bool(getattr(args, "list_incomplete", False))


def _result_task(args: Any) -> str | None:
    command = args.command
    if isinstance(command, ResultCommandArgs):
        return command.task
    return cast("str | None", getattr(args, "result_task", None))


def _logs_task(args: Any) -> str | None:
    command = args.command
    if isinstance(command, LogsCommandArgs):
        return command.task
    return cast("str | None", getattr(args, "logs_task", None))


def _logs_job_id(args: Any) -> str | None:
    command = args.command
    if isinstance(command, LogsCommandArgs):
        return command.job_id
    return cast("str | None", getattr(args, "logs_job_id", None))


def _logs_job(args: Any) -> bool:
    command = args.command
    if isinstance(command, LogsCommandArgs):
        return command.job
    return bool(getattr(args, "logs_job", False))


def _logs_list(args: Any) -> bool:
    command = args.command
    if isinstance(command, LogsCommandArgs):
        return command.list
    return bool(getattr(args, "logs_list", False))


def _logs_all(args: Any) -> bool:
    command = args.command
    if isinstance(command, LogsCommandArgs):
        return command.all
    return bool(getattr(args, "logs_all", False))


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
    named_tasks = experiment.tasks()
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

    work_graph = build_work_graph(set(experiment.tasks().values()))
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
    command_name = _command_name(args.command)
    match command_name:
        case "run":
            executor = _resolve_executor(args)
            workspace = _resolve_workspace(args)
            run_task = _run_task(args)
            run_tui = _run_tui(args)
            if run_task is None and run_tui:
                tui.submit_and_watch_jobs(experiment=args.experiment, executor=executor, workspace=workspace)
            elif run_task is None and not run_tui:
                tasks = set(args.experiment.tasks().values())
                executor.submit(tasks=tasks, workspace=workspace, blocking=True)
            else:
                task_name = _resolve_command_task(command=command_name, task_name=run_task)
                args.experiment[task_name].submit(executor=executor, workspace=workspace, blocking=True)
        case "count":
            workspace = _resolve_workspace(args)
            complete_count, total_count = _count_completion(args.experiment.tasks(), workspace)
            console.print(_format_count_message(complete_count=complete_count, total_count=total_count))
        case "tree" | "incomplete":
            workspace = _resolve_workspace(args)
            all_named_tasks = args.experiment.tasks()
            tree_task_arg = _tree_task(args) if command_name == "tree" else None
            if tree_task_arg is not None:
                if tree_task_arg not in all_named_tasks:
                    msg = f"Experiment has no task named {tree_task_arg!r}. Known tasks: {sorted(all_named_tasks)}"
                    raise ValueError(msg)
                named_tasks: Mapping[str, Task[Any]] = {tree_task_arg: all_named_tasks[tree_task_arg]}
            else:
                named_tasks = all_named_tasks
            complete_count, total_count = _count_completion(named_tasks, workspace)
            incomplete_only = command_name == "incomplete" or _tree_incomplete(args)
            if incomplete_only and complete_count == total_count:
                console.print("[green]No incomplete tasks.[/green]")
                return
            tree = _build_task_tree(
                named_tasks,
                workspace,
                title="Incomplete Tasks" if incomplete_only else "Tasks",
                show_all=_tree_all(args),
                max_depth=_tree_max_depth(args),
                cacheable_only=_tree_cacheable_only(args),
                incomplete_only=incomplete_only,
            )
            console.print(tree)
        case "list":
            workspace = _resolve_workspace(args)
            complete_count, total_count = _count_completion(args.experiment.tasks(), workspace)
            list_incomplete = _list_incomplete(args)
            if list_incomplete and complete_count == total_count:
                console.print("[green]No incomplete tasks.[/green]")
                return

            title = "Incomplete Tasks" if list_incomplete else "Tasks"
            lines = _build_task_list_lines(
                args.experiment.tasks(),
                workspace,
                cacheable_only=_list_cacheable_only(args),
                incomplete_only=list_incomplete,
            )

            if not lines:
                console.print("[yellow]No tasks matched filters.[/yellow]")
                return

            console.print(_styled_title(title))
            for line in lines:
                console.print(line)
        case "result":
            workspace = _resolve_workspace(args)
            task_key = _resolve_command_task(command=command_name, task_name=_result_task(args))
            console.print(Pretty(args.experiment.result(task_key, workspace=workspace)))
        case "logs":
            workspace = _resolve_workspace(args)
            task_query = _logs_task(args)
            job_id = _logs_job_id(args)
            job_mode = _logs_job(args)
            list_mode = _logs_list(args)
            show_all = _logs_all(args)

            if not job_mode:
                # Task log mode.
                if task_query is not None:
                    task = _resolve_task_or_hash(args.experiment, task_query)
                    if list_mode:
                        status = _status_indicator(done=_task_done(task, workspace))
                        console.print(f"{status} {_task_display_label(task)}")
                        _list_task_logs(task, workspace, console)
                    else:
                        try:
                            with workspace.open_task_log(task, mode="r", job_id=job_id) as f:
                                log_content = f.read()
                        except FileNotFoundError:
                            console.print("[dim]No logs found.[/dim]")
                            return
                        status = _status_indicator(done=_task_done(task, workspace))
                        console.rule(f"{status} {_task_display_label(task)}")
                        console.print(Text.from_ansi(log_content))
                else:
                    tasks = _iter_task_closure(args.experiment.tasks().values())
                    printed_any = False
                    for task in tasks:
                        if list_mode:
                            status = _status_indicator(done=_task_done(task, workspace))
                            console.print(f"{status} {_task_display_label(task)}")
                            _list_task_logs(task, workspace, console)
                            printed_any = True
                        else:
                            try:
                                with workspace.open_task_log(task, mode="r", job_id=job_id) as f:
                                    log_content = f.read()
                            except FileNotFoundError:
                                continue
                            status = _status_indicator(done=_task_done(task, workspace))
                            console.rule(f"{status} {_task_display_label(task)}")
                            console.print(Text.from_ansi(log_content))
                            printed_any = True
                    if not printed_any:
                        console.print("[dim]No logs found.[/dim]")
            # Job log mode.
            elif task_query is not None:
                task = _resolve_task_or_hash(args.experiment, task_query)
                work_unit = _find_work_unit_for_task(args.experiment, task)
                log_paths = sorted(workspace.job_log_iter(work_unit), key=lambda p: p.stat().st_mtime)

                if job_id is not None:
                    log_paths = [p for p in log_paths if f"_{job_id}.log" in p.name]

                if not log_paths:
                    console.print("[dim]No job logs found.[/dim]")
                    return

                if list_mode:
                    _list_job_logs(log_paths, console)
                else:
                    logs_to_show = log_paths if show_all else log_paths[-1:]
                    for log_path in logs_to_show:
                        _print_log_content(
                            log_path, console, rule_title=f"[bright_white]{escape(log_path.name)}[/bright_white]"
                        )
            else:
                log_paths = sorted(workspace.job_log_iter(), key=lambda p: p.stat().st_mtime)
                if not log_paths:
                    console.print("[dim]No job logs found.[/dim]")
                    return

                if list_mode:
                    _list_job_logs(log_paths, console)
                else:
                    logs_to_show = log_paths if show_all else log_paths[-1:]
                    for log_path in logs_to_show:
                        _print_log_content(
                            log_path, console, rule_title=f"[bright_white]{escape(log_path.name)}[/bright_white]"
                        )


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


def experiment_cli(experiment_cls: type[Any], argv: list[str] | tuple[str, ...] | None = None) -> None:
    """Parse CLI args and execute experiment command.

    Args:
        experiment_cls: Experiment class type to expose on CLI.
        argv: Optional command tokens; defaults to ``sys.argv[1:]`` when ``None``.

    Notes:
        Parsing happens in two phases so executor/workspace concrete argument
        schemas can be selected dynamically from ``--executor``/``--workspace`` flags.
    """
    from misen.executor import Executor
    from misen.workspace import Workspace

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

    cli_fields: list[tuple[Any, ...]] = [
        ("experiment", tyro.conf.OmitArgPrefixes[experiment_cls]),  # ty:ignore[invalid-type-form]
    ]
    if bootstrap_args.executor != "auto":
        cli_fields.append(("executor", Executor.resolve_type(bootstrap_args.executor)))
    if bootstrap_args.workspace != "auto":
        cli_fields.append(("workspace", Workspace.resolve_type(bootstrap_args.workspace)))
    cli_fields.extend(
        [
            ("command", ExperimentCommandArgs, field(default_factory=command_default_factories[resolved_command])),
            ("config", Path | None, bootstrap_args.config),
            ("executor", ExecutorType | str, bootstrap_args.executor),
            ("workspace", WorkspaceType | str, bootstrap_args.workspace),
        ]
    )

    console = Console()
    parsed = tyro.cli(make_dataclass("", cli_fields), args=args_list)
    _execute_command(args=cast("Any", parsed), console=console)
