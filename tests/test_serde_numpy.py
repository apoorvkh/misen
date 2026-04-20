"""Round-trip tests for ``DictOfNdarraysSerializer``.

Focused on the dict-dispatch path — the single-ndarray and masked-array
serializers are covered implicitly by being the long-standing baseline.
"""
# ruff: noqa: D103, S101

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

from misen.utils import serde
from misen.utils.serde import UnserializableTypeError

if TYPE_CHECKING:
    import pathlib

np = pytest.importorskip("numpy")

Roundtrip = Callable[[Any], Any]


@pytest.fixture
def roundtrip(tmp_path: pathlib.Path) -> Roundtrip:
    counter = {"n": 0}

    def _do(obj: Any) -> Any:
        counter["n"] += 1
        d = tmp_path / f"data_{counter['n']}"
        d.mkdir()
        serde.save(obj, d)
        return serde.load(d)

    return _do


def _meta(directory: pathlib.Path) -> dict[str, Any]:
    return json.loads((directory / "serde_meta.json").read_text())


def test_dict_of_ndarrays_roundtrip(roundtrip: Roundtrip) -> None:
    original = {
        "x": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "y": np.array([1.5, 2.5, 3.5]),
    }
    loaded = roundtrip(original)
    assert type(loaded) is dict
    assert list(loaded) == list(original)
    for k, v in original.items():
        np.testing.assert_array_equal(loaded[k], v)
        assert loaded[k].dtype == v.dtype


def test_dict_of_ndarrays_uses_npz(tmp_path: pathlib.Path) -> None:
    d = tmp_path / "n"
    d.mkdir()
    serde.save({"a": np.zeros(2)}, d)
    meta = _meta(d)
    assert meta["serializer"].endswith(".DictOfNdarraysSerializer")
    assert meta["container"] == "dict"
    assert (d / "data.npz").exists()


def test_ordered_dict_of_ndarrays_roundtrip(roundtrip: Roundtrip) -> None:
    original = OrderedDict([("b", np.array([1])), ("a", np.array([2, 3]))])
    loaded = roundtrip(original)
    assert type(loaded) is OrderedDict
    assert list(loaded) == ["b", "a"]
    for k, v in original.items():
        np.testing.assert_array_equal(loaded[k], v)


def test_empty_dict_falls_through_to_msgpack(tmp_path: pathlib.Path) -> None:
    """Empty dicts never write a .npz file — they fall through to msgpack."""
    d = tmp_path / "e"
    d.mkdir()
    serde.save({}, d)
    meta = _meta(d)
    assert meta["serializer"].endswith(".MsgpackSerializer")
    assert not (d / "data.npz").exists()


def test_mixed_dict_raises_cleanly(roundtrip: Roundtrip) -> None:
    with pytest.raises(UnserializableTypeError):
        roundtrip({"arr": np.array([1, 2]), "scalar": 1.5})


def test_non_str_keyed_dict_raises_cleanly(roundtrip: Roundtrip) -> None:
    with pytest.raises(UnserializableTypeError):
        roundtrip({1: np.array([1, 2])})


def test_masked_array_dict_does_not_match(roundtrip: Roundtrip) -> None:
    """MaskedArray values must not be silently downcast via np.savez.

    The dispatch path should skip the dict-of-ndarrays serializer (whose
    match requires plain ndarrays), and the msgpack fallback should then
    raise cleanly rather than drop the mask.
    """
    mask = np.ma.MaskedArray([1, 2, 3], mask=[False, True, False])
    with pytest.raises(UnserializableTypeError):
        roundtrip({"m": mask})


def test_many_arrays_preserve_insertion_order(roundtrip: Roundtrip) -> None:
    # Regression: np.savez preserves kwargs insertion order and npz.files
    # reads them back in the same order.  If this ever changes, key order
    # would silently drift on load.
    keys = [f"k{i}" for i in range(20)]
    original = {k: np.array([int(k[1:])]) for k in keys}
    loaded = roundtrip(original)
    assert list(loaded) == keys
