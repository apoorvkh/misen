"""Round-trip tests for ``DictOfJaxArraysSerializer``."""
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

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


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


def test_dict_of_jax_arrays_roundtrip(roundtrip: Roundtrip) -> None:
    original = {"w": jnp.arange(6).reshape(2, 3), "b": jnp.zeros(3)}
    loaded = roundtrip(original)
    assert type(loaded) is dict
    assert list(loaded) == list(original)
    for k, v in original.items():
        assert bool(jnp.array_equal(loaded[k], v))


def test_dict_of_jax_arrays_uses_dict_serializer(tmp_path: pathlib.Path) -> None:
    d = tmp_path / "j"
    d.mkdir()
    serde.save({"a": jnp.zeros(2)}, d)
    meta = _meta(d)
    assert meta["serializer"].endswith(".DictOfJaxArraysSerializer")
    assert meta["container"] == "dict"
    assert (d / "data.npz").exists()


def test_ordered_dict_of_jax_arrays_roundtrip(roundtrip: Roundtrip) -> None:
    original = OrderedDict([("second", jnp.ones(2)), ("first", jnp.zeros(3))])
    loaded = roundtrip(original)
    assert type(loaded) is OrderedDict
    assert list(loaded) == ["second", "first"]


def test_mixed_dict_raises_cleanly(roundtrip: Roundtrip) -> None:
    with pytest.raises(UnserializableTypeError):
        roundtrip({"arr": jnp.array([1, 2]), "scalar": 1.5})
