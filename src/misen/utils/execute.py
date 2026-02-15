"""CLI entrypoint for executing a payload file."""

from pathlib import Path

import cloudpickle
import tyro


def execute(payload: Path) -> None:
    """Execute a payload file (containing a cloudpickle-serialized function)."""
    payload_fn = cloudpickle.loads(payload.read_bytes())
    payload_fn()


if __name__ == "__main__":
    tyro.cli(execute)
