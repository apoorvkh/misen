"""Import-boundary regression tests."""
# ruff: noqa: D103

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["misen", "misen.cli"])
def test_core_import_does_not_load_command_implementations(module: str) -> None:
    code = (
        f"import {module}, sys\n"
        "assert 'libcst' not in sys.modules\n"
        "assert 'sky' not in sys.modules\n"
        "assert 'misen.executors.skypilot' not in sys.modules\n"
        "assert 'misen.utils.cli.fill' not in sys.modules\n"
        "assert 'misen.utils.cli.experiment' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603


def test_fill_command_import_is_callable() -> None:
    code = (
        "import misen.utils.cli, sys\n"
        "assert 'libcst' not in sys.modules\n"
        "from misen.utils.cli.fill import fill\n"
        "assert callable(fill)\n"
        "assert 'libcst' in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603


def test_skypilot_executor_alias_does_not_import_optional_sdk() -> None:
    code = (
        "import sys\n"
        "from misen.executor import Executor\n"
        "resolved = Executor.resolve_type('skypilot')\n"
        "assert resolved.__name__ == 'SkyPilotExecutor'\n"
        "assert 'misen.executors.skypilot' in sys.modules\n"
        "assert 'sky' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
