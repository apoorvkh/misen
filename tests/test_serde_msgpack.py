"""Round-trip tests for msgpack-tagged stdlib types.

These tests are the executable contract for the tagged msgpack encoder
in ``serde/libs/stdlib.py``.  Each parametrized case asserts that
:func:`misen.utils.serde.save` / :func:`misen.utils.serde.load`
round-trip a value with both type and value fidelity:

    type(loaded) is type(original) and loaded == original

The error cases assert that values the serializer cannot represent
surface as :class:`SerializationError` rather than crashing with
``msgspec``'s raw ``TypeError``/``OverflowError``.
"""
# ruff: noqa: D103, S101, PLR2004

from __future__ import annotations

import dataclasses
import datetime
import enum
import math
import pathlib
import re
import types
import zoneinfo
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable
from datetime import timezone
from decimal import Decimal
from fractions import Fraction
from ipaddress import (
    IPv4Address,
    IPv4Interface,
    IPv4Network,
    IPv6Address,
    IPv6Interface,
    IPv6Network,
)
from typing import Any, NamedTuple
from uuid import UUID

import pytest

from misen.exceptions import SerializationError
from misen.utils import serde

UTC = timezone.utc

Roundtrip = Callable[[Any], Any]


@pytest.fixture
def roundtrip(tmp_path: pathlib.Path) -> Roundtrip:
    """Save *obj* via :func:`serde.save` and load it back from a temp directory."""
    counter = {"n": 0}

    def _do(obj: Any) -> Any:
        counter["n"] += 1
        d = tmp_path / f"data_{counter['n']}"
        d.mkdir()
        serde.save(obj, d)
        return serde.load(d)

    return _do


def _assert_strict_roundtrip(original: Any, loaded: Any) -> None:
    """Assert ``loaded`` matches ``original`` in both type and value."""
    assert type(loaded) is type(original), (
        f"type mismatch: original={type(original).__name__} loaded={type(loaded).__name__}"
    )
    assert loaded == original, f"value mismatch: original={original!r} loaded={loaded!r}"


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        1,
        -1,
        2**62,
        -(2**63),
        2**64 - 1,
        0.0,
        -0.0,
        3.14159,
        1e308,
        float("inf"),
        -float("inf"),
        "",
        "hello",
        "unicode: 日本語 \u200b",
        b"",
        b"hello",
        b"\x00\x01\xff",
    ],
)
def test_primitive_roundtrip(roundtrip: Roundtrip, value: Any) -> None:
    _assert_strict_roundtrip(value, roundtrip(value))


def test_nan_roundtrip(roundtrip: Roundtrip) -> None:
    loaded = roundtrip(float("nan"))
    assert type(loaded) is float
    assert math.isnan(loaded)


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        [],
        [1, 2, 3],
        [None, True, "x", b"y", 1.5],
        [[1, 2], [3, 4]],
        (),
        (1, 2, 3),
        (1, "two", 3.0),
        ((1, 2), (3, 4)),
        set(),
        {1, 2, 3},
        {"a", "b"},
        frozenset(),
        frozenset([1, 2, 3]),
        {},
        {"a": 1, "b": 2},
        {1: "one", 2: "two"},
        {"nested": {"a": [1, 2]}},
    ],
)
def test_container_roundtrip(roundtrip: Roundtrip, value: Any) -> None:
    _assert_strict_roundtrip(value, roundtrip(value))


def test_tuple_dict_key_roundtrip(roundtrip: Roundtrip) -> None:
    """msgspec.msgpack handles tuple keys natively when not tagged."""
    value = {(1, 2): "a", (3, 4): "b"}
    loaded = roundtrip(value)
    assert loaded == value
    for key in loaded:
        assert type(key) is tuple


def test_bytes_dict_key_roundtrip(roundtrip: Roundtrip) -> None:
    value = {b"key": "v"}
    loaded = roundtrip(value)
    assert loaded == value
    assert type(next(iter(loaded))) is bytes


def test_escaped_dict_collision(roundtrip: Roundtrip) -> None:
    """A user dict whose key collides with our tag convention round-trips."""
    value = {"__t": "looks_like_a_tag", "v": [1, 2, 3]}
    _assert_strict_roundtrip(value, roundtrip(value))


def test_nested_tuple_in_dict(roundtrip: Roundtrip) -> None:
    value = {"a": (1, 2), "b": [(3, 4), (5, 6)]}
    _assert_strict_roundtrip(value, roundtrip(value))


# ---------------------------------------------------------------------------
# Stdlib value types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        bytearray(b""),
        bytearray(b"hello"),
        complex(0, 0),
        complex(1.5, -2.5),
        datetime.datetime(2024, 1, 1, 12, 30, 45, tzinfo=UTC),
        datetime.datetime(2024, 6, 15, 8, 0, microsecond=123456, tzinfo=UTC),
        datetime.date(2024, 1, 1),
        datetime.date(1, 1, 1),
        datetime.time(0, 0),
        datetime.time(23, 59, 59, 999999),
        datetime.timedelta(),
        datetime.timedelta(days=1, hours=2, minutes=3, seconds=4),
        datetime.timedelta(days=-5),
        UUID("00000000-0000-0000-0000-000000000000"),
        UUID("12345678-1234-5678-1234-567812345678"),
        Decimal(0),
        Decimal("3.14159265358979323846"),
        Decimal("-1e-10"),
        Fraction(0),
        Fraction(1, 3),
        Fraction(-7, 11),
        range(0),
        range(10),
        range(1, 10, 2),
        range(-5, 5),
        slice(None),
        slice(10),
        slice(1, 10, 2),
        slice(None, None, -1),
        re.compile("foo"),
        re.compile(r"\d+", re.IGNORECASE | re.MULTILINE),
        re.compile(rb"\d+"),
        zoneinfo.ZoneInfo("UTC"),
        zoneinfo.ZoneInfo("America/New_York"),
        IPv4Address("1.2.3.4"),
        IPv6Address("::1"),
        IPv4Network("192.168.0.0/24"),
        IPv6Network("2001:db8::/32"),
        IPv4Interface("10.0.0.1/8"),
        IPv6Interface("fe80::1/64"),
    ],
)
def test_stdlib_value_roundtrip(roundtrip: Roundtrip, value: Any) -> None:
    _assert_strict_roundtrip(value, roundtrip(value))


def test_naive_datetime_roundtrip(roundtrip: Roundtrip) -> None:
    """Naive datetimes (no tzinfo) round-trip with type and value preserved."""
    naive = datetime.datetime(2024, 1, 1, 12, 30, 45)  # noqa: DTZ001
    loaded = roundtrip(naive)
    assert type(loaded) is datetime.datetime
    assert loaded == naive
    assert loaded.tzinfo is None


def test_datetime_fold_preserved(roundtrip: Roundtrip) -> None:
    original = datetime.datetime(2023, 11, 5, 1, 30, fold=1)  # noqa: DTZ001
    loaded = roundtrip(original)
    assert type(loaded) is datetime.datetime
    assert loaded == original
    assert loaded.fold == 1


def test_time_fold_preserved(roundtrip: Roundtrip) -> None:
    original = datetime.time(1, 30, fold=1)
    loaded = roundtrip(original)
    assert type(loaded) is datetime.time
    assert loaded.fold == 1


@pytest.mark.parametrize(
    "value",
    [
        pathlib.PurePosixPath("/usr/local/bin"),
        pathlib.PureWindowsPath(r"C:\Users\x"),
        pathlib.PosixPath("/usr/local/bin"),
    ],
)
def test_path_roundtrip(roundtrip: Roundtrip, value: pathlib.PurePath) -> None:
    _assert_strict_roundtrip(value, roundtrip(value))


# ---------------------------------------------------------------------------
# Nested / collection helpers
# ---------------------------------------------------------------------------


def test_simple_namespace_roundtrip(roundtrip: Roundtrip) -> None:
    ns = types.SimpleNamespace(a=1, b="two", c=[1, 2, 3])
    loaded = roundtrip(ns)
    assert type(loaded) is types.SimpleNamespace
    assert vars(loaded) == vars(ns)


def test_ordered_dict_roundtrip(roundtrip: Roundtrip) -> None:
    od = OrderedDict([("a", 1), ("b", 2), ("c", 3)])
    loaded = roundtrip(od)
    assert type(loaded) is OrderedDict
    assert list(loaded.items()) == list(od.items())


def test_deque_roundtrip(roundtrip: Roundtrip) -> None:
    dq = deque([1, 2, 3], maxlen=5)
    loaded = roundtrip(dq)
    assert type(loaded) is deque
    assert list(loaded) == [1, 2, 3]
    assert loaded.maxlen == 5


def test_deque_no_maxlen(roundtrip: Roundtrip) -> None:
    dq = deque([1, 2, 3])
    loaded = roundtrip(dq)
    assert type(loaded) is deque
    assert loaded.maxlen is None


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class _Color(enum.Enum):
    RED = "r"
    GREEN = "g"


class _Priority(enum.IntEnum):
    LOW = 1
    HIGH = 2


def test_enum_roundtrip(roundtrip: Roundtrip) -> None:
    _assert_strict_roundtrip(_Color.RED, roundtrip(_Color.RED))


def test_int_enum_roundtrip(roundtrip: Roundtrip) -> None:
    _assert_strict_roundtrip(_Priority.LOW, roundtrip(_Priority.LOW))


# ---------------------------------------------------------------------------
# NamedTuple and dataclass
# ---------------------------------------------------------------------------


class _Point(NamedTuple):
    x: int
    y: int


def test_namedtuple_roundtrip(roundtrip: Roundtrip) -> None:
    p = _Point(1, 2)
    loaded = roundtrip(p)
    assert type(loaded) is _Point
    assert loaded == p


@dataclasses.dataclass
class _Inner:
    x: int
    label: str


@dataclasses.dataclass
class _Outer:
    inner: _Inner
    items: list[int]


def test_dataclass_roundtrip(roundtrip: Roundtrip) -> None:
    obj = _Inner(x=5, label="hi")
    _assert_strict_roundtrip(obj, roundtrip(obj))


def test_nested_dataclass_roundtrip(roundtrip: Roundtrip) -> None:
    """The nested dataclass keeps its type — it does not become a dict."""
    obj = _Outer(inner=_Inner(x=5, label="hi"), items=[1, 2, 3])
    loaded = roundtrip(obj)
    assert type(loaded) is _Outer
    assert type(loaded.inner) is _Inner
    assert loaded == obj


def test_bytes_roundtrip(roundtrip: Roundtrip) -> None:
    loaded = roundtrip(b"hello world")
    assert type(loaded) is bytes
    assert loaded == b"hello world"


def test_bytearray_roundtrip(roundtrip: Roundtrip) -> None:
    loaded = roundtrip(bytearray(b"hello world"))
    assert type(loaded) is bytearray
    assert loaded == bytearray(b"hello world")


# ---------------------------------------------------------------------------
# Error cases — must raise SerializationError, not crash with raw msgspec errors
# ---------------------------------------------------------------------------


def test_big_int_raises_clean_error(roundtrip: Roundtrip) -> None:
    with pytest.raises(SerializationError):
        roundtrip(2**100)


def test_int_subclass_raises_clean_error(roundtrip: Roundtrip) -> None:
    class MyInt(int):
        pass

    with pytest.raises(SerializationError):
        roundtrip(MyInt(5))


def test_str_subclass_raises_clean_error(roundtrip: Roundtrip) -> None:
    class MyStr(str):
        __slots__ = ()

    with pytest.raises(SerializationError):
        roundtrip(MyStr("hi"))


def test_dict_subclass_raises_clean_error(roundtrip: Roundtrip) -> None:
    dd: defaultdict[str, list[int]] = defaultdict(list)
    dd["a"].append(1)
    with pytest.raises(SerializationError):
        roundtrip(dd)


def test_unknown_class_raises_clean_error(roundtrip: Roundtrip) -> None:
    class Custom:
        pass

    with pytest.raises(SerializationError):
        roundtrip(Custom())


def test_big_int_inside_container_raises_clean_error(roundtrip: Roundtrip) -> None:
    with pytest.raises(SerializationError):
        roundtrip([1, 2, 2**100])
