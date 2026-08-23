"""Regression tests for stable hashing's public exception contract."""

# ruff: noqa: D103, S101

import pytest

from misen import HashError
from misen.utils.hashing import stable_hash
from misen.utils.hashing.libs.stdlib import ZoneInfoHandler


def test_zoneinfo_without_stable_key_raises_hash_error() -> None:
    with pytest.raises(HashError, match="stable key"):
        ZoneInfoHandler.digest(object(), lambda _value: 0)


def test_released_memoryview_raises_hash_error_with_cause() -> None:
    value = memoryview(b"payload")
    value.release()

    with pytest.raises(HashError, match="released memoryview") as raised:
        stable_hash(value)

    assert isinstance(raised.value.__cause__, ValueError)
