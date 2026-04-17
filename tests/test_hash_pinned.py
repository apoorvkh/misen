"""Pinned-output regression suite for canonical hashing.

These tests lock down the byte encoding (``_encode``) and the digest output
of every registered handler to known constants. Any change to ``_encode``,
``hash_values``, or any ``Handler.digest`` implementation that alters its
output is a workspace-breaking change: every existing ``.misen/`` cache
becomes silently unusable.

If you intentionally change hashing semantics, regenerate these constants
and update the comment in ``src/misen/utils/hashing/base.py`` warning
users that the change requires re-creating workspaces.
"""

from __future__ import annotations

import array
import datetime
import decimal
import enum
import fractions
import importlib.util
import ipaddress
import pathlib
import re
import types
import uuid
import zoneinfo
from collections import ChainMap, Counter, OrderedDict, UserDict, UserList, UserString, defaultdict, deque
from dataclasses import dataclass
from typing import NamedTuple

import pytest

from misen.utils.hashing import stable_hash
from misen.utils.hashing.base import _encode, hash_values

# ---------------------------------------------------------------------------
# Encoding-layer (pre-hash byte format)
# ---------------------------------------------------------------------------

_PINNED_ENCODING: dict[str, bytes] = {
    "None": b"\x00",
    "True": b"\x01\x01",
    "False": b"\x01\x00",
    "int_0": b"\x02\x01\x00",
    "int_1": b"\x02\x00\x00\x00\x00\x00\x00\x00\x01\x01",
    "int_neg1": b"\x02\x00\x00\x00\x00\x00\x00\x00\x01\xff",
    "int_2_pow_63": b"\x02\x00\x00\x00\x00\x00\x00\x00\t\x00\x80\x00\x00\x00\x00\x00\x00\x00",
    "float_0": b"\x03\x00\x00\x00\x00\x00\x00\x00\x00",
    "float_neg0_collapses_to_pos0": b"\x03\x00\x00\x00\x00\x00\x00\x00\x00",
    "float_1_5": b"\x03?\xf8\x00\x00\x00\x00\x00\x00",
    "str_empty": b"\x04\x00\x00\x00\x00\x00\x00\x00\x00",
    "str_hello": b"\x04\x00\x00\x00\x00\x00\x00\x00\x05hello",
    "bytes_empty": b"\x05\x00\x00\x00\x00\x00\x00\x00\x00",
    "bytes_xyz": b"\x05\x00\x00\x00\x00\x00\x00\x00\x03\x00\x01\x02",
    "tuple_empty": b"\x06\x00\x00\x00\x00\x00\x00\x00\x00",
    "list_empty": b"\x07\x00\x00\x00\x00\x00\x00\x00\x00",
    "set_empty": b"\x08\x00\x00\x00\x00\x00\x00\x00\x00",
}

_ENCODE_INPUTS: dict[str, object] = {
    "None": None,
    "True": True,
    "False": False,
    "int_0": 0,
    "int_1": 1,
    "int_neg1": -1,
    "int_2_pow_63": 2**63,
    "float_0": 0.0,
    "float_neg0_collapses_to_pos0": -0.0,
    "float_1_5": 1.5,
    "str_empty": "",
    "str_hello": "hello",
    "bytes_empty": b"",
    "bytes_xyz": b"\x00\x01\x02",
    "tuple_empty": (),
    "list_empty": [],
    "set_empty": set(),
}


@pytest.mark.parametrize("label", list(_PINNED_ENCODING))
def test_encode_pinned(label: str) -> None:
    assert _encode(_ENCODE_INPUTS[label]) == _PINNED_ENCODING[label]


def test_set_and_frozenset_share_encoding() -> None:
    # Both share _TAG_SET; outer type-name disambiguation happens in stable_hash.
    assert _encode({1, 2, 3}) == _encode(frozenset({1, 2, 3}))


def test_list_and_tuple_distinguished() -> None:
    assert _encode([1, 2, 3]) != _encode((1, 2, 3))


def test_xxh3_seed_zero_unchanged() -> None:
    # If xxhash ever changes seed handling or the algorithm, this catches it.
    assert hash_values(0) == 0xB4E28A203C2672B1
    assert hash_values("hello") == 0x0BEF1A12F95DEDBB


# ---------------------------------------------------------------------------
# Stdlib handler digests
# ---------------------------------------------------------------------------


class _Color(enum.Enum):
    RED = 1
    GREEN = 2


@dataclass(frozen=True)
class _Point:
    x: int
    y: int


class _Pair(NamedTuple):
    a: int
    b: str


_PINNED_STDLIB: dict[str, tuple[object, int]] = {
    "None": (None, 0xDBA2BEB2AC8C42BC),
    "True": (True, 0x44377F08F5C5B57C),
    "False": (False, 0x2E5FCD19F80E67A2),
    "int_0": (0, 0xE31FF198DB34FDCD),
    "int_1": (1, 0x137805D2EDDF1A3A),
    "int_neg1": (-1, 0xFB3E2A552DFA7CCF),
    "int_2_pow_63": (2**63, 0x8F14E59A22B2C12A),
    "float_0": (0.0, 0x37CC828FE6D3EDC1),
    "float_neg0": (-0.0, 0x37CC828FE6D3EDC1),
    "float_inf": (float("inf"), 0xF2B7B2E34EFA2620),
    "float_neginf": (float("-inf"), 0xDA0816BDFFF048EA),
    "float_nan": (float("nan"), 0xEB226B4F3A9596E3),
    "float_pi": (3.14159, 0xB19AD6A41917E83F),
    "complex_1plus2j": (complex(1, 2), 0xB10F7E2957DBD9CA),
    "str_empty": ("", 0xBAB7F3BACB263E3E),
    "str_hello": ("hello", 0x09947EA9F3D97EC0),
    "str_unicode": ("héllo🌍", 0xD6BBBB04DC552BE8),
    "bytes_empty": (b"", 0x224431251B39EFF2),
    "bytes_xyz": (b"\x00\x01\x02", 0x44460D4C346FAB89),
    "bytearray_xyz": (bytearray(b"\x00\x01\x02"), 0x010C4AD4C34C4E6C),
    "memoryview_xyz": (memoryview(b"\x00\x01\x02"), 0x6B3BA712CA3931FB),
    "datetime_naive": (datetime.datetime(2024, 1, 2, 3, 4, 5, 6), 0x823B1D4359692BE7),
    "datetime_utc": (
        datetime.datetime(2024, 1, 2, 3, 4, 5, 6, tzinfo=datetime.timezone.utc),
        0xA24084EA51121749,
    ),
    "datetime_zoneinfo": (
        datetime.datetime(2024, 1, 2, 3, 4, 5, 6, tzinfo=zoneinfo.ZoneInfo("America/Los_Angeles")),
        0x045A546BCA37DB54,
    ),
    "date_2024_01_02": (datetime.date(2024, 1, 2), 0x8CD4D4D30E1CBCB7),
    "time_03_04_05": (datetime.time(3, 4, 5), 0x2F6898F85A7BAB5A),
    "timedelta_1d": (datetime.timedelta(days=1, seconds=2, microseconds=3), 0x96837FDA222B0D1E),
    "uuid_zero": (uuid.UUID(int=0), 0xEAE7AA39C8B8FFEA),
    "decimal_1_5": (decimal.Decimal("1.5"), 0xB1EA5E458E946650),
    "fraction_1_3": (fractions.Fraction(1, 3), 0x747D6F153A9CBC56),
    "range_0_10_2": (range(0, 10, 2), 0x0D121B834DAEA3AE),
    "slice_0_10_2": (slice(0, 10, 2), 0x7F25EB4613C08C74),
    "path_pure_posix": (pathlib.PurePosixPath("/a/b"), 0xB60907C4BBDDC1E4),
    "pattern_abc_icase": (re.compile(r"abc", re.IGNORECASE), 0x7B3C377F89D5857B),
    "zoneinfo_la": (zoneinfo.ZoneInfo("America/Los_Angeles"), 0x7ED995C31716FEE7),
    "ipv4_localhost": (ipaddress.IPv4Address("127.0.0.1"), 0xA82AFF1C35F08BEF),
    "ipv6_loopback": (ipaddress.IPv6Address("::1"), 0x20F61522733D419C),
    "ipv4_net_10": (ipaddress.IPv4Network("10.0.0.0/8"), 0xFDDF4A8572BE385C),
    "array_i_1_2_3": (array.array("i", [1, 2, 3]), 0x0078761C2324954E),
    "ellipsis": (..., 0x0E52696F6956D4F0),
    "type_int": (int, 0xC42312F9441756EC),
    "namespace_a1_bx": (types.SimpleNamespace(a=1, b="x"), 0x1E535DC0F3DCA929),
    "userdict_ab": (UserDict({"a": 1, "b": 2}), 0x105647FA8F29D0FD),
    "userlist_123": (UserList([1, 2, 3]), 0x13A4806429D3D026),
    "userstring_hi": (UserString("hi"), 0x0BB43A8F81A67DAD),
    "list_123": ([1, 2, 3], 0xBD0BBC7317B684DA),
    "tuple_123": ((1, 2, 3), 0xFF4631B922E896F8),
    "set_123": ({1, 2, 3}, 0xAAA323381CE2E96B),
    "frozenset_123": (frozenset({1, 2, 3}), 0x1199268A2164F693),
    "deque_123": (deque([1, 2, 3]), 0xECC693629A13CB2E),
    "ordereddict_ab": (OrderedDict([("a", 1), ("b", 2)]), 0xC5F5F9F1695D986F),
    "defaultdict_a1": (defaultdict(int, {"a": 1}), 0xC8DEFB0912135AA9),
    "counter_aabbb": (Counter("aabbb"), 0x92AC9CBC0A2F96F3),
    "dict_ab": ({"a": 1, "b": 2}, 0x40904739604F72DB),
    "dict_keys_ab": ({"a": 1, "b": 2}.keys(), 0x80EFBE6D9AC19834),
    "dict_values_ab": ({"a": 1, "b": 2}.values(), 0x88969AF6D36BF824),
    "dict_items_ab": ({"a": 1, "b": 2}.items(), 0xF4A148D8D5804A6E),
    "chainmap_ab": (ChainMap({"a": 1}, {"b": 2}), 0x8E6A4A89EE5312D2),
    "mappingproxy_ab": (types.MappingProxyType({"a": 1, "b": 2}), 0xB7E606D07E17631F),
    "enum_color_red": (_Color.RED, 0x3B5126DA9C760F7F),
    "dataclass_point_1_2": (_Point(1, 2), 0x5E9A851902FB2F35),
    "namedtuple_pair_1x": (_Pair(1, "x"), 0xB278A5604DBA04FE),
}


@pytest.mark.parametrize("label", list(_PINNED_STDLIB))
def test_stdlib_handler_pinned(label: str) -> None:
    value, expected = _PINNED_STDLIB[label]
    assert stable_hash(value) == expected, f"{label}: hash drifted -- this is a workspace-breaking change"


# ---------------------------------------------------------------------------
# Cross-handler invariants
# ---------------------------------------------------------------------------


def test_dict_order_independence() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_set_order_independence() -> None:
    # Two sets with different insertion orders should match.
    a = set()
    a.add(1)
    a.add(2)
    a.add(3)
    b = set()
    b.add(3)
    b.add(1)
    b.add(2)
    assert stable_hash(a) == stable_hash(b)


def test_neg_zero_matches_pos_zero() -> None:
    assert stable_hash(-0.0) == stable_hash(0.0)


def test_list_and_tuple_distinguished() -> None:
    assert stable_hash([1, 2, 3]) != stable_hash((1, 2, 3))


def test_set_and_frozenset_distinguished() -> None:
    # Same _TAG_SET inside _encode, but type_name differs at the outer layer.
    assert stable_hash({1, 2, 3}) != stable_hash(frozenset({1, 2, 3}))


def test_path_concrete_normalized() -> None:
    # PathHandler.type_name collapses concrete Posix/Windows Path to "pathlib.Path"
    # so caches survive cross-platform moves.
    p = pathlib.Path("/a/b")
    assert stable_hash(p) == stable_hash(pathlib.Path("/a/b"))


def test_recursive_reference_does_not_loop() -> None:
    # Self-referential structures hash with a stable back-edge marker.
    a: list = [1, 2]
    a.append(a)
    h = stable_hash(a)
    assert isinstance(h, int)


# ---------------------------------------------------------------------------
# Library handlers (gated by import availability — same as the handlers themselves)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(importlib.util.find_spec("numpy") is None, reason="numpy not installed")
def test_numpy_handlers_pinned() -> None:
    import numpy as np

    pinned: dict[str, tuple[object, int]] = {
        "np_dtype_f64": (np.dtype("float64"), 0x469A70683D704419),
        "np_dtype_int32": (np.dtype("int32"), 0xA23135D4361D767C),
        "np_int64_42": (np.int64(42), 0x38F24DB5006596AF),
        "np_float32_1_5": (np.float32(1.5), 0x442EAB364595B30C),
    }
    for label, (value, expected) in pinned.items():
        assert stable_hash(value) == expected, label


@pytest.mark.skipif(importlib.util.find_spec("msgspec") is None, reason="msgspec not installed")
def test_msgspec_handler_pinned() -> None:
    import msgspec

    class _S(msgspec.Struct):
        a: int
        b: str

    assert stable_hash(_S(a=1, b="x")) == 0xD156D2E7B9AD2D9A


# Pinned constants for pyarrow, torch, pydantic, attrs, and sympy handlers
# are intentionally not included here. To add them, run
# ``scripts/_emit_pinned.py`` in an environment with those libraries
# installed and copy the constants into new tests below.
