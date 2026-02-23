"""CLI entrypoint for executing serialized work-unit payloads."""

from pathlib import Path

import cloudpickle
import tyro

from misen.utils.resource_binding import apply_resource_binding_from_env


def execute(payload: Path) -> None:
    """Execute a cloudpickle payload file.

    Args:
        payload: Path to payload bytes representing a zero-argument callable.
    """
    apply_resource_binding_from_env()
    payload_fn = cloudpickle.loads(payload.read_bytes())
    payload_fn()


if __name__ == "__main__":
    tyro.cli(execute)
