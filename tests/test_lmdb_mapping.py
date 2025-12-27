from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import pytest

from misen.task import Hash
from misen.workspaces.disk import LMDBMapping

if TYPE_CHECKING:
    from pathlib import Path


def make_db_path(tmp_path: Path) -> Path:
    # LMDBMapping uses subdir=False => pass a file path, not a directory.
    return tmp_path / "test.lmdb"


@pytest.fixture
def m(tmp_path: Path):
    M = LMDBMapping[Hash, Hash]
    inst = M(make_db_path(tmp_path))
    try:
        yield inst
    finally:
        inst.env.close()


def test_must_be_parameterized(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match=r"Construct as LMDBMapping\[KeyType, ValueType\]"):
        LMDBMapping(make_db_path(tmp_path))  # type: ignore[call-arg]


def test_class_getitem_binds_types(tmp_path: Path) -> None:
    Bound = LMDBMapping[Hash, Hash]
    assert Bound._key_type is Hash
    assert Bound._value_type is Hash

    inst = Bound(make_db_path(tmp_path))
    try:
        assert inst._key_type is Hash
        assert inst._value_type is Hash
    finally:
        inst.env.close()


def test_set_get_contains_len_and_overwrite(m) -> None:
    assert len(m) == 0
    assert Hash(1) not in m

    m[Hash(1)] = Hash(10)
    assert len(m) == 1
    assert Hash(1) in m
    assert m[Hash(1)] == Hash(10)
    assert isinstance(m[Hash(1)], Hash)

    # overwrite should not increase entry count
    m[Hash(1)] = Hash(11)
    assert len(m) == 1
    assert m[Hash(1)] == Hash(11)


def test_getitem_missing_raises_keyerror(m) -> None:
    with pytest.raises(KeyError):
        _ = m[Hash(999)]


def test_delitem_and_del_missing(m) -> None:
    m[Hash(7)] = Hash(70)
    assert len(m) == 1

    del m[Hash(7)]
    assert len(m) == 0
    assert Hash(7) not in m

    with pytest.raises(KeyError):
        del m[Hash(7)]


def test_iter_yields_hashes_in_numeric_order(m) -> None:
    # Fixed-width big-endian means lexicographic == numeric order.
    m[Hash(2)] = Hash(20)
    m[Hash(1)] = Hash(10)
    m[Hash(3)] = Hash(30)

    keys = list(m)
    assert keys == [Hash(1), Hash(2), Hash(3)]
    assert all(isinstance(k, Hash) for k in keys)


def test_contains_rejects_wrong_type(m) -> None:
    m[Hash(1)] = Hash(100)

    # plain int is NOT instance of Hash, so __contains__ should return False
    assert 1 not in m
    assert "1" not in m
    assert object() not in m

    assert Hash(1) in m


def test_persistence_across_instances(tmp_path: Path) -> None:
    db_path = make_db_path(tmp_path)
    M = LMDBMapping[Hash, Hash]

    a = M(db_path)
    try:
        a[Hash(123)] = Hash(456)
        assert a[Hash(123)] == Hash(456)
    finally:
        a.env.close()

    b = M(db_path)
    try:
        assert len(b) == 1
        assert Hash(123) in b
        assert b[Hash(123)] == Hash(456)
    finally:
        b.env.close()


def test_roundtrip_boundaries(tmp_path: Path) -> None:
    M = LMDBMapping[Hash, Hash]
    inst = M(make_db_path(tmp_path))
    try:
        inst[Hash(0)] = Hash(0)
        inst[Hash(2**64 - 1)] = Hash(2**64 - 1)

        assert inst[Hash(0)] == Hash(0)
        assert inst[Hash(2**64 - 1)] == Hash(2**64 - 1)
    finally:
        inst.env.close()


def test_dict_and_items_view(m) -> None:
    m[Hash(1)] = Hash(10)
    m[Hash(2)] = Hash(20)

    # dict(...) uses iteration over keys + __getitem__
    d = dict(m)
    assert d == {Hash(1): Hash(10), Hash(2): Hash(20)}

    # items/keys/values come from MutableMapping mixin
    assert set(m.keys()) == {Hash(1), Hash(2)}
    assert set(m.values()) == {Hash(10), Hash(20)}
    assert set(m.items()) == {(Hash(1), Hash(10)), (Hash(2), Hash(20))}


def test_update_from_dict_and_iterable_pairs(m) -> None:
    m.update({Hash(1): Hash(10), Hash(2): Hash(20)})
    assert dict(m) == {Hash(1): Hash(10), Hash(2): Hash(20)}

    # update from iterable of pairs
    m.update([(Hash(2), Hash(200)), (Hash(3), Hash(30))])
    assert dict(m) == {Hash(1): Hash(10), Hash(2): Hash(200), Hash(3): Hash(30)}


def test_setdefault_and_pop(m) -> None:
    # setdefault
    out = m.setdefault(Hash(5), Hash(50))
    assert out == Hash(50)
    assert m[Hash(5)] == Hash(50)

    # setdefault on existing key doesn't overwrite
    out2 = m.setdefault(Hash(5), Hash(500))
    assert out2 == Hash(50)
    assert m[Hash(5)] == Hash(50)

    # pop
    v = m.pop(Hash(5))
    assert v == Hash(50)
    assert Hash(5) not in m

    with pytest.raises(KeyError):
        m.pop(Hash(5))

    # pop with default
    assert m.pop(Hash(999), Hash(123)) == Hash(123)


def test_clear_via_mutablemapping_default_impl(m) -> None:
    m.update({Hash(1): Hash(10), Hash(2): Hash(20), Hash(3): Hash(30)})
    assert len(m) == 3

    m.clear()
    assert len(m) == 0
    assert dict(m) == {}


def test_iter_then_mutate_does_not_crash(m):
    m.update([(Hash(1), Hash(10)), (Hash(2), Hash(20)), (Hash(3), Hash(30))])
    it = iter(m)
    first = next(it)
    m[Hash(4)] = Hash(40)  # write during active read cursor
    del m[Hash(2)]
    rest = list(it)  # should not raise  # noqa: F841
    assert first in [Hash(1), Hash(2), Hash(3)]


def test_hash_encode_bounds():
    with pytest.raises(OverflowError):
        Hash(2**64).encode()
    with pytest.raises(OverflowError):
        Hash(-1).encode()


def test_persistence_across_interpreter_sessions(tmp_path: Path) -> None:
    db_path = tmp_path / "session_test.lmdb"

    writer = textwrap.dedent(
        r"""
        from pathlib import Path
        import sys

        from misen.task import Hash
        from misen.workspaces.disk import LMDBMapping

        db_path = Path(sys.argv[1])

        m = LMDBMapping[Hash, Hash](db_path)
        try:
            m[Hash(1)] = Hash(10)
            m[Hash(2)] = Hash(20)
            m[Hash(2)] = Hash(200)  # overwrite
            assert len(m) == 2
        finally:
            m.env.close()
        """
    )

    reader = textwrap.dedent(
        r"""
        from pathlib import Path
        import sys

        from misen.task import Hash
        from misen.workspaces.disk import LMDBMapping

        db_path = Path(sys.argv[1])

        m = LMDBMapping[Hash, Hash](db_path)
        try:
            assert len(m) == 2
            assert m[Hash(1)] == Hash(10)
            assert m[Hash(2)] == Hash(200)
            # sanity: iteration order should be numeric
            assert list(m) == [Hash(1), Hash(2)]
        finally:
            m.env.close()
        """
    )

    subprocess.run([sys.executable, "-c", writer, str(db_path)], check=True)
    subprocess.run([sys.executable, "-c", reader, str(db_path)], check=True)
