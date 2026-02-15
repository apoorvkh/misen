"""Experiment CLI helpers."""

from __future__ import annotations

from dataclasses import make_dataclass
from pathlib import Path
from typing import Any, Literal, cast

import tyro

from misen.utils.settings import DEFAULT_SETTINGS_FILE, Settings


def experiment_cli(experiment_cls: type[Any]) -> None:
    """Parse CLI args and run the requested experiment command."""
    from misen.executor import Executor, ExecutorType
    from misen.workspace import Workspace, WorkspaceType

    fields_without_defaults: list[tuple[Any, ...]] = []
    fields_with_defaults = [
        ("command", Literal["run", "count"], "run"),
        ("settings_file", Path, DEFAULT_SETTINGS_FILE),
        ("executor_type", ExecutorType | Literal["auto"], "auto"),
        ("workspace_type", WorkspaceType | Literal["auto"], "auto"),
    ]

    args, _ = tyro.cli(
        make_dataclass("", fields_without_defaults + fields_with_defaults),
        add_help=False,
        return_unknown_args=True,
    )
    args = cast("Any", args)

    if args.executor_type != "auto":
        fields_without_defaults.append(("executor", Executor.resolve_type(args.executor_type)))
    if args.workspace_type != "auto":
        fields_without_defaults.append(("workspace", Workspace.resolve_type(args.workspace_type)))

    fields_without_defaults.append(("experiment", tyro.conf.OmitArgPrefixes[experiment_cls]))  # ty:ignore[invalid-type-form]

    args = tyro.cli(make_dataclass("", fields_without_defaults + fields_with_defaults))
    args = cast("Any", args)

    settings = Settings(file=args.settings_file)
    executor = Executor.auto(settings=settings) if args.executor_type == "auto" else args.executor
    workspace = Workspace.auto(settings=settings) if args.workspace_type == "auto" else args.workspace

    match args.command:
        case "run":
            args.experiment.run(executor=executor, workspace=workspace)
