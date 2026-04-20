"""Round-trip tests for ``TorchTensorSerializer`` and ``DictOfTensorsSerializer``.

These tests assert that the on-disk save/load contract holds for:

* single ``torch.Tensor`` values via ``torch.save`` / ``torch.load(weights_only=True)``;
* flat ``dict[str, torch.Tensor]`` and ``OrderedDict[str, torch.Tensor]`` —
  container type and insertion order preserved by pickle;
* content-sensitive dispatch: dicts mixing tensors with non-tensor values or
  using non-str keys do NOT hit the dict-of-tensors path and must either
  fall back to msgpack (and round-trip cleanly) or raise
  :class:`UnserializableTypeError`.

``pytest.importorskip`` gates the module — if torch isn't installed, the
entire file is skipped in CI (matching the project's existing pattern for
optional-lib tests).
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

torch = pytest.importorskip("torch")


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


# ---------------------------------------------------------------------------
# Single tensor
# ---------------------------------------------------------------------------


def test_tensor_roundtrip(roundtrip: Roundtrip) -> None:
    original = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    loaded = roundtrip(original)
    assert isinstance(loaded, torch.Tensor)
    assert loaded.dtype == original.dtype
    assert loaded.shape == original.shape
    assert torch.equal(loaded, original)


def test_tensor_int_dtype_roundtrip(roundtrip: Roundtrip) -> None:
    original = torch.tensor([1, 2, 3], dtype=torch.int64)
    loaded = roundtrip(original)
    assert loaded.dtype == torch.int64
    assert torch.equal(loaded, original)


def test_tensor_uses_tensor_serializer(tmp_path: pathlib.Path) -> None:
    d = tmp_path / "t"
    d.mkdir()
    serde.save(torch.zeros(2), d)
    meta = _meta(d)
    assert meta["serializer"].endswith(".TorchTensorSerializer")
    assert (d / "data.pt").exists()


# ---------------------------------------------------------------------------
# Dict of tensors
# ---------------------------------------------------------------------------


def test_dict_of_tensors_roundtrip(roundtrip: Roundtrip) -> None:
    original = {
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "bias": torch.zeros(3),
    }
    loaded = roundtrip(original)
    assert type(loaded) is dict
    # Pickle preserves insertion order.
    assert list(loaded) == list(original)
    for k, v in original.items():
        assert torch.equal(loaded[k], v)


def test_dict_of_tensors_uses_dict_serializer(tmp_path: pathlib.Path) -> None:
    d = tmp_path / "t"
    d.mkdir()
    serde.save({"a": torch.zeros(2)}, d)
    meta = _meta(d)
    assert meta["serializer"].endswith(".DictOfTensorsSerializer")
    assert (d / "data.pt").exists()


def test_ordered_dict_of_tensors_roundtrip(roundtrip: Roundtrip) -> None:
    """Pickle preserves both the ``OrderedDict`` type and its insertion order."""
    original = OrderedDict([("z", torch.ones(2)), ("a", torch.zeros(3))])
    loaded = roundtrip(original)
    assert type(loaded) is OrderedDict
    assert list(loaded) == ["z", "a"]
    for k, v in original.items():
        assert torch.equal(loaded[k], v)


def test_state_dict_from_nn_module_roundtrip(roundtrip: Roundtrip) -> None:
    """Round-trip an ``nn.Module`` state_dict.

    ``nn.Module.state_dict()`` returns an OrderedDict of tensors — the most
    important real-world case this serializer exists for.
    """
    model = torch.nn.Linear(4, 2)
    sd = model.state_dict()
    assert type(sd) is OrderedDict  # sanity check torch's API shape
    loaded = roundtrip(sd)
    assert type(loaded) is OrderedDict
    assert set(loaded) == set(sd)
    for k, v in sd.items():
        assert torch.equal(loaded[k], v)


# ---------------------------------------------------------------------------
# Dispatch: what is and isn't a dict-of-tensors
# ---------------------------------------------------------------------------


def test_empty_dict_falls_through_to_msgpack(tmp_path: pathlib.Path) -> None:
    """Empty dict shouldn't write a binary file — the msgpack fallback wins."""
    d = tmp_path / "e"
    d.mkdir()
    serde.save({}, d)
    meta = _meta(d)
    assert meta["serializer"].endswith(".MsgpackSerializer")
    assert not (d / "data.pt").exists()


def test_mixed_dict_raises_cleanly(roundtrip: Roundtrip) -> None:
    """Mixed dicts (tensor + primitive) must raise, not partially serialize.

    The dict-of-tensors match requires all values be tensors, so dispatch
    falls to MsgpackSerializer, which cannot encode a tensor and raises
    UnserializableTypeError.
    """
    with pytest.raises(UnserializableTypeError):
        roundtrip({"tensor": torch.zeros(2), "scalar": 1.5})


def test_non_str_keyed_dict_raises_cleanly(roundtrip: Roundtrip) -> None:
    """Non-str keys must not hit the dict-of-tensors path.

    Disqualifying via the ``match`` check means msgpack is picked next,
    which then encounters the tensor value and raises.
    """
    with pytest.raises(UnserializableTypeError):
        roundtrip({1: torch.zeros(2)})


def test_dict_of_primitives_still_msgpack(tmp_path: pathlib.Path) -> None:
    """Ordinary dicts must still route to MsgpackSerializer.

    Regression check on the volatile_types change — removing the fast-path
    ``dict`` entry from the by-type map could easily break this.
    """
    d = tmp_path / "p"
    d.mkdir()
    serde.save({"a": 1, "b": [1, 2]}, d)
    meta = _meta(d)
    assert meta["serializer"].endswith(".MsgpackSerializer")


def test_nn_module_still_uses_module_serializer(tmp_path: pathlib.Path) -> None:
    """DictOfTensorsSerializer's match must not accidentally swallow Modules."""
    d = tmp_path / "m"
    d.mkdir()
    model = torch.nn.Linear(2, 2)
    serde.save(model, d)
    meta = _meta(d)
    assert meta["serializer"].endswith(".TorchModuleSerializer")


def test_dispatch_does_not_cache_dict_contents(tmp_path: pathlib.Path) -> None:
    """Interleaved dict saves must dispatch by contents, not by cached type.

    Regression guard on ``volatile_types``: if ``dict`` were ever removed
    from the volatile set, the first dict-of-tensors save would cache
    ``DictOfTensorsSerializer`` against ``type(obj) is dict`` and every
    subsequent dict save would wrongly get the same serializer.
    """
    tensor_dir = tmp_path / "t"
    tensor_dir.mkdir()
    serde.save({"w": torch.zeros(2)}, tensor_dir)
    assert _meta(tensor_dir)["serializer"].endswith(".DictOfTensorsSerializer")

    primitive_dir = tmp_path / "p"
    primitive_dir.mkdir()
    serde.save({"a": 1}, primitive_dir)
    assert _meta(primitive_dir)["serializer"].endswith(".MsgpackSerializer")
