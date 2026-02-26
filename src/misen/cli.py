"""Top-level ``misen`` CLI entrypoint.

This module intentionally stays thin; command logic lives in
``misen.utils.cli``.
"""

from __future__ import annotations

import tyro

from misen.utils.cli import fill

__all__ = []


def main(argv: list[str] | None = None) -> int:
    """Execute the ``misen`` CLI."""
    return int(tyro.extras.subcommand_cli_from_dict({"fill": fill}, args=argv))


if __name__ == "__main__":
    raise SystemExit(main())
