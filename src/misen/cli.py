"""Top-level ``misen`` CLI entrypoint.

This module intentionally stays thin; command logic lives in
``misen.utils.cli``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Annotated, Any, cast

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
            help="Experiment class reference in '<module>:<ExperimentClass>' or '<path.py>:<ExperimentClass>' format.",
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


def main(argv: list[str] | None = None) -> int:
    """Execute the ``misen`` CLI."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list or args_list[0] in {"-h", "--help"}:
        try:
            tyro.cli(cast("Any", TopLevelCommand), args=["--help"])
        except SystemExit as exc:
            return _system_exit_code(exc)
        return 0

    parsed, unknown_args = cast(
        "tuple[object, list[str]]",
        tyro.cli(
            cast("Any", TopLevelCommand),
            args=args_list,
            return_unknown_args=True,
            add_help=False,
        ),
    )

    if isinstance(parsed, _ExperimentSelection):
        return experiment(argv=[parsed.reference, *unknown_args])

    if isinstance(parsed, _FillSelection):
        return int(tyro.cli(fill, args=unknown_args))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
