# ruff: noqa: D103, EM101, S101, TRY003
"""Tests for exception-preserving cleanup."""

import pytest

from misen.utils.cleanup import _cleanup_on_exit


def test_cleanup_failure_surfaces_after_success() -> None:
    def fail_cleanup() -> None:
        raise RuntimeError("cleanup failed")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        with _cleanup_on_exit(fail_cleanup, "test cleanup"):
            pass


def test_cleanup_failure_does_not_replace_active_failure() -> None:
    error = ValueError("body failed")

    def fail_cleanup() -> None:
        raise RuntimeError("cleanup failed")

    with pytest.raises(ValueError, match="body failed") as raised:
        with _cleanup_on_exit(fail_cleanup, "test cleanup"):
            raise error

    assert raised.value is error
    assert raised.value.__notes__ == ["Additionally, test cleanup failed: RuntimeError: cleanup failed"]
