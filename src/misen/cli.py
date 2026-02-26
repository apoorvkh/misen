"""Top-level ``misen`` CLI entrypoint.

This module intentionally stays thin; command logic lives in
``misen.utils.cli``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Annotated

import tyro

from misen.utils.cli.experiment import experiment
from misen.utils.cli.fill import fill

__all__ = []


@dataclass(frozen=True)
class _FillSelection:
    pass


@dataclass(frozen=True)
class _ExperimentSelection:
    reference: str


def _select_fill() -> _FillSelection:
    """Fill missing ``@task(id=...)`` values in Python files."""
    return _FillSelection()


def _select_experiment(
    reference: Annotated[
        str,
        tyro.conf.Positional,
        tyro.conf.arg(
            name="experiment-ref",
            help="Experiment class reference in '<module>:<ExperimentClass>' format.",
        ),
    ],
) -> _ExperimentSelection:
    """Run an experiment CLI from a class reference."""
    return _ExperimentSelection(reference=reference)


TopLevelCommand = (
    Annotated[_FillSelection, tyro.conf.subcommand(name="fill", constructor=_select_fill)]
    | Annotated[_ExperimentSelection, tyro.conf.subcommand(name="experiment", constructor=_select_experiment)]
    | Annotated[None, tyro.conf.Suppress]
)


def main(argv: list[str] | None = None) -> int:
    """Execute the ``misen`` CLI."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list or args_list[0] in {"-h", "--help"}:
        try:
            tyro.cli(TopLevelCommand, args=["--help"])
        except SystemExit as exc:
            return int(exc.code)
        return 0

    parsed, unknown_args = tyro.cli(
        TopLevelCommand,
        args=args_list,
        return_unknown_args=True,
        add_help=False,
    )

    if isinstance(parsed, _ExperimentSelection):
        return experiment(argv=[parsed.reference, *unknown_args])

    if isinstance(parsed, _FillSelection):
        return int(tyro.cli(fill, args=unknown_args))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
