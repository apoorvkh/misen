"""CLI entrypoint for executing serialized work-unit payloads."""

from pathlib import Path

import cloudpickle
import tyro


def execute(payload: Path) -> None:
    """Execute a cloudpickle payload file.

    Args:
        payload: Path to payload bytes representing a zero-argument callable.
    """
    payload_fn = cloudpickle.loads(payload.read_bytes())
    payload_fn()


if __name__ == "__main__":
    tyro.cli(execute)
