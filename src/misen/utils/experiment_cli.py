"""Helpers for generating experiment CLIs with ``tyro``."""

from __future__ import annotations

from dataclasses import make_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast, get_args

import tyro
from rich.console import Console
from rich.markup import escape
from rich.pretty import Pretty
from rich.tree import Tree

from misen.utils import tui
from misen.utils.runtime_events import task_label
from misen.utils.settings import DEFAULT_SETTINGS_FILE, Settings

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from misen.tasks import Task


ExperimentCommand = Literal["run", "list", "tree", "count", "result", "incomplete"]
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


def _resolve_command_task(*, command: str, task_name: str | None) -> str:
    """Resolve task selector for command-specific task operations."""
    if task_name is None:
        msg = f"{command!r} command requires a task name."
        raise ValueError(msg)
    return task_name


def _task_sort_key(task: Task[Any]) -> tuple[str, str]:
    """Return stable sort key for task display."""
    return (task.properties.id, task.task_hash().b32())


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


def _split_hash_suffix(label: str) -> tuple[str, str | None]:
    """Split trailing `` [HASH]`` suffix from a task label."""
    if not label.endswith("]"):
        return label, None

    prefix, separator, remainder = label.rpartition(" [")
    if not separator or not remainder.endswith("]"):
        return label, None

    hash_suffix = remainder[:-1]
    return (prefix, hash_suffix) if hash_suffix else (label, None)


def _split_call_signature(label: str) -> tuple[str, str | None]:
    """Split ``name(args)`` text into name and args."""
    if "(" not in label or not label.endswith(")"):
        return label, None
    name, _, arg_tail = label.partition("(")
    return name, arg_tail[:-1]


def _styled_title(title: str) -> str:
    """Return consistently-styled title text for list/tree outputs."""
    color = "yellow" if "Incomplete" in title else "cyan"
    return f"[bold {color}]{escape(title)}[/bold {color}]"


def _task_display_label(task: Task[Any]) -> str:
    """Return one task label with CLI-specific cache marker placement."""
    label = task_label(task, include_arguments=True)
    label_without_hash, hash_suffix = _split_hash_suffix(label)
    task_name, task_args = _split_call_signature(label_without_hash)

    styled_label = f"[bright_white]{escape(task_name)}[/bright_white]"
    if task_args is not None:
        styled_label += f"[dim]({escape(task_args)})[/dim]"

    if task.properties.cache:
        styled_label += " [cyan][C][/cyan]"
    if hash_suffix is not None:
        styled_label += f" [blue][{escape(hash_suffix)}][/blue]"

    return styled_label


def _build_task_tree(
    named_tasks: Mapping[str, Task[Any]],
    workspace: Any,
    *,
    title: str = "Tasks",
    show_all: bool = False,
    max_depth: int | None = None,
    cacheable_only: bool = False,
    incomplete_only: bool = False,
) -> tuple[Tree, bool]:
    """Build a rich tree for experiment tasks and dependencies."""
    if max_depth is not None and max_depth < 0:
        msg = "Tree max depth must be >= 0."
        raise ValueError(msg)

    tree = Tree(_styled_title(title))
    rendered: set[Task[Any]] = set()
    used_shared_marker = False

    def include_task(task: Task[Any]) -> bool:
        if cacheable_only and not task.properties.cache:
            return False
        return not (incomplete_only and _task_done(task, workspace))

    def add_task(
        branch: Tree,
        task: Task[Any],
        *,
        name: str | None,
        ancestry: set[Task[Any]],
        depth: int,
    ) -> None:
        if not include_task(task):
            return

        nonlocal used_shared_marker
        status = _status_indicator(done=_task_done(task, workspace))
        name_prefix = f"[bold magenta]{escape(name)}[/bold magenta]: " if name is not None else ""
        line = f"{status} {name_prefix}{_task_display_label(task)}"

        if task in ancestry:
            branch.add(f"{line} [red](cycle)[/red]")
            return
        if not show_all and task in rendered:
            used_shared_marker = True
            branch.add(f"{line} [dim](*)[/dim]")
            return

        child_ancestry = set(ancestry)
        child_ancestry.add(task)
        visible_dependencies: list[Task[Any]] = []
        skipped_shared_dependency = False
        for dependency in sorted(task.dependencies, key=_task_sort_key):
            if not include_task(dependency):
                continue
            if not show_all and dependency in rendered and dependency not in child_ancestry:
                used_shared_marker = True
                skipped_shared_dependency = True
                continue
            visible_dependencies.append(dependency)

        if skipped_shared_dependency:
            line = f"{line} [dim](*)[/dim]"

        rendered.add(task)
        child_branch = branch.add(line)

        if max_depth is not None and depth >= max_depth:
            return

        for dependency in visible_dependencies:
            add_task(
                child_branch,
                dependency,
                name=None,
                ancestry=child_ancestry,
                depth=depth + 1,
            )

    rendered_any = False
    for name, task in sorted(named_tasks.items()):
        if not include_task(task):
            continue
        add_task(tree, task, name=name, ancestry=set(), depth=0)
        rendered_any = True

    if not rendered_any:
        tree.add("[dim]No tasks matched filters.[/dim]")
    return tree, used_shared_marker and not show_all


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
        if cacheable_only and not task.properties.cache:
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

    settings = Settings(file=args.settings_file)
    return Workspace.auto(settings=settings) if args.workspace_type == "auto" else args.workspace


def _resolve_executor(args: Any) -> Any:
    """Resolve executor instance from parsed CLI args."""
    from misen.executor import Executor

    settings = Settings(file=args.settings_file)
    return Executor.auto(settings=settings) if args.executor_type == "auto" else args.executor


def _execute_command(*, args: Any, console: Console) -> None:
    """Execute resolved experiment CLI command."""
    match args.command:
        case "run":
            executor = _resolve_executor(args)
            workspace = _resolve_workspace(args)
            run_task = args.run_task
            run_tui = args.run_tui
            if run_task is None and run_tui:
                tui.submit_and_watch_jobs(experiment=args.experiment, executor=executor, workspace=workspace)
            elif run_task is None and not run_tui:
                tasks = set(args.experiment.tasks().values())
                executor.submit(tasks=tasks, workspace=workspace, blocking=True)
            else:
                task_name = _resolve_command_task(command=args.command, task_name=run_task)
                args.experiment[task_name].submit(executor=executor, workspace=workspace, blocking=True)
        case "count":
            workspace = _resolve_workspace(args)
            complete_count, total_count = _count_completion(args.experiment.tasks(), workspace)
            console.print(_format_count_message(complete_count=complete_count, total_count=total_count))
        case "tree" | "incomplete":
            workspace = _resolve_workspace(args)
            complete_count, total_count = _count_completion(args.experiment.tasks(), workspace)
            incomplete_only = args.command == "incomplete" or args.tree_incomplete
            if incomplete_only and complete_count == total_count:
                console.print("[green]No incomplete tasks.[/green]")
                return
            tree, has_shared_dependencies = _build_task_tree(
                args.experiment.tasks(),
                workspace,
                title="Incomplete Tasks" if incomplete_only else "Tasks",
                show_all=args.tree_all,
                max_depth=args.tree_max_depth,
                cacheable_only=args.tree_cacheable_only,
                incomplete_only=incomplete_only,
            )
            console.print(
                tree
            )
            if has_shared_dependencies:
                console.print("[dim](*) Dependencies already listed.[/dim]")
        case "list":
            workspace = _resolve_workspace(args)
            complete_count, total_count = _count_completion(args.experiment.tasks(), workspace)
            if args.list_incomplete and complete_count == total_count:
                console.print("[green]No incomplete tasks.[/green]")
                return

            title = "Incomplete Tasks" if args.list_incomplete else "Tasks"
            lines = _build_task_list_lines(
                args.experiment.tasks(),
                workspace,
                cacheable_only=args.list_cacheable_only,
                incomplete_only=args.list_incomplete,
            )

            if not lines:
                console.print("[yellow]No tasks matched filters.[/yellow]")
                return

            console.print(_styled_title(title))
            for line in lines:
                console.print(line)
        case "result":
            workspace = _resolve_workspace(args)
            task_key = _resolve_command_task(command=args.command, task_name=args.result_task)
            console.print(Pretty(args.experiment.result(task_key, workspace=workspace)))


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


def _command_specific_fields(command: ExperimentCommand) -> list[tuple[Any, ...]]:
    """Return command-specific CLI fields to inject on second parse."""
    match command:
        case "run":
            return [
                ("run_task", tyro.conf.Positional[str | None], None),
                ("run_tui", RunTuiArg, True),
            ]
        case "tree":
            return [
                ("tree_all", TreeAllArg, False),
                ("tree_max_depth", TreeDepthArg, None),
                ("tree_cacheable_only", TreeCacheableOnlyArg, False),
                ("tree_incomplete", TreeIncompleteArg, False),
            ]
        case "list":
            return [
                ("list_cacheable_only", ListCacheableOnlyArg, False),
                ("list_incomplete", ListIncompleteArg, False),
            ]
        case "incomplete":
            return [
                ("tree_all", TreeAllArg, False),
                ("tree_max_depth", TreeDepthArg, None),
                ("tree_cacheable_only", TreeCacheableOnlyArg, False),
                ("tree_incomplete", TreeIncompleteArg, True),
            ]
        case "result":
            return [("result_task", tyro.conf.Positional[str | None], None)]
        case _:
            return []


def experiment_cli(experiment_cls: type[Any]) -> None:
    """Parse CLI args and execute experiment command.

    Args:
        experiment_cls: Experiment class type to expose on CLI.

    Notes:
        Parsing happens in two phases so executor/workspace concrete argument
        schemas can be selected dynamically from ``*_type`` flags.
    """
    from misen.executor import Executor, ExecutorType
    from misen.workspace import Workspace, WorkspaceType

    console = Console()

    fields_without_defaults: list[tuple[Any, ...]] = []
    fields_with_defaults = [
        ("command", tyro.conf.Positional[str | None], None),
        ("settings_file", Path, DEFAULT_SETTINGS_FILE),
        ("executor_type", ExecutorType | Literal["auto"], "auto"),
        ("workspace_type", WorkspaceType | Literal["auto"], "auto"),
    ]

    args, unknown_args = tyro.cli(
        make_dataclass("", fields_without_defaults + fields_with_defaults),
        add_help=False,
        return_unknown_args=True,
    )
    args = cast("Any", args)
    command = _resolve_command(command_token=args.command, unknown_args=unknown_args)

    if args.executor_type != "auto":
        fields_without_defaults.append(("executor", Executor.resolve_type(args.executor_type)))
    if args.workspace_type != "auto":
        fields_without_defaults.append(("workspace", Workspace.resolve_type(args.workspace_type)))

    command_specific_fields = _command_specific_fields(command)

    fields_without_defaults.append(("experiment", tyro.conf.OmitArgPrefixes[experiment_cls]))  # ty:ignore[invalid-type-form]
    second_pass_fields = [
        *fields_without_defaults,
        ("command", tyro.conf.Positional[ExperimentCommand], command),
        *command_specific_fields,
        ("settings_file", Path, DEFAULT_SETTINGS_FILE),
        ("executor_type", ExecutorType | Literal["auto"], "auto"),
        ("workspace_type", WorkspaceType | Literal["auto"], "auto"),
    ]

    args = tyro.cli(make_dataclass("", second_pass_fields))
    args = cast("Any", args)

    _execute_command(args=args, console=console)
