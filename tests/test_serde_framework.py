"""Spike tests for the recursive, batching serde design (``serde``).

Four things these tests demonstrate beyond the v1 flat design:

1. **Batched nested tensors** — a dict-of-dict-of-tensors produces a
   single ``tensors.pt`` (v1 would fall through to msgpack here and
   fail, or would need a DictOfDictOfTensorsSerializer).
2. **Mixed containers** — dicts mixing tensors with primitives round
   trip naturally; v1 raises :class:`UnserializableTypeError`.
3. **Shared-object identity** — the same tensor referenced three times
   is written once and decodes to one Python object.
4. **Directory escape hatch** — ``nn.Module`` still works via
   :class:`DirectorySerializer`, showing the batching model coexists
   with bespoke subdirectory layouts.
"""
# ruff: noqa: D103, S101

from __future__ import annotations

import json
from collections import OrderedDict
from typing import TYPE_CHECKING

import pytest

from misen.utils import serde

if TYPE_CHECKING:
    import pathlib

torch = pytest.importorskip("torch")


def _manifest(directory: pathlib.Path) -> dict:
    return json.loads((directory / serde.MANIFEST_FILENAME).read_text())


# ---------------------------------------------------------------------------
# Parity with v1: single tensor, flat dict, OrderedDict state_dict
# ---------------------------------------------------------------------------


def test_flat_tensor_roundtrip(tmp_path: pathlib.Path) -> None:
    original = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    serde.save(original, tmp_path)
    loaded = serde.load(tmp_path)
    assert isinstance(loaded, torch.Tensor)
    assert torch.equal(loaded, original)


def test_flat_dict_of_tensors_roundtrip(tmp_path: pathlib.Path) -> None:
    original = {
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "bias": torch.zeros(3),
    }
    serde.save(original, tmp_path)
    loaded = serde.load(tmp_path)
    assert type(loaded) is dict
    assert list(loaded) == list(original)
    for k, v in original.items():
        assert torch.equal(loaded[k], v)


def test_ordered_dict_type_preserved(tmp_path: pathlib.Path) -> None:
    original = OrderedDict([("z", torch.ones(2)), ("a", torch.zeros(3))])
    serde.save(original, tmp_path)
    loaded = serde.load(tmp_path)
    assert type(loaded) is OrderedDict
    assert list(loaded) == ["z", "a"]


def test_state_dict_roundtrip(tmp_path: pathlib.Path) -> None:
    """Real-world case: nn.Module state_dict is an OrderedDict of tensors."""
    model = torch.nn.Linear(4, 2)
    sd = model.state_dict()
    assert type(sd) is OrderedDict
    serde.save(sd, tmp_path)
    loaded = serde.load(tmp_path)
    assert type(loaded) is OrderedDict
    assert set(loaded) == set(sd)
    for k, v in sd.items():
        assert torch.equal(loaded[k], v)


# ---------------------------------------------------------------------------
# Design wins
# ---------------------------------------------------------------------------


def test_nested_tensors_batched_into_single_file(tmp_path: pathlib.Path) -> None:
    """v2 WIN: nested dict of tensors packs into ONE tensors.pt.

    v1 can only batch at the top level (DictOfTensorsSerializer hits
    the whole dict).  v2 recurses and batches all tensors in the tree
    into one ``torch.save`` call regardless of nesting depth.
    """
    original = {
        "encoder": {"w": torch.ones(2), "b": torch.zeros(3)},
        "decoder": {"w": torch.ones(5), "b": torch.zeros(1)},
    }
    serde.save(original, tmp_path)

    tensors_blob = tmp_path / "leaves" / "torch_tensor" / "tensors.pt"
    assert tensors_blob.exists(), "All tensors should be in a single batched blob"

    loaded = serde.load(tmp_path)
    for section in ("encoder", "decoder"):
        for k in ("w", "b"):
            assert torch.equal(loaded[section][k], original[section][k])


def test_mixed_dict_now_works(tmp_path: pathlib.Path) -> None:
    """v2 WIN: mixed container (primitives + tensors) round trips.

    v1 raises UnserializableTypeError here because the whole dict
    goes to MsgpackSerializer which can't encode a Tensor value.
    v2 dispatches each child independently, so primitives route to
    the msgpack batch and the tensor routes to the tensor batch.
    """
    original = {"lr": 0.001, "step": 100, "weights": torch.ones(4)}
    serde.save(original, tmp_path)

    loaded = serde.load(tmp_path)
    assert loaded["lr"] == pytest.approx(0.001)
    assert loaded["step"] == 100
    assert torch.equal(loaded["weights"], original["weights"])


def test_shared_tensor_is_written_once_and_decodes_as_same_object(
    tmp_path: pathlib.Path,
) -> None:
    """v2 WIN: id()-keyed memo preserves shared-leaf identity."""
    shared = torch.ones(1000)
    original = {"a": shared, "b": shared, "c": {"nested": shared}}
    serde.save(original, tmp_path)

    manifest = _manifest(tmp_path)
    # The three references should all carry the same leaf id.
    root = manifest["root"]
    leaf_id_a = root["children"]["a"]["id"]
    leaf_id_b = root["children"]["b"]["id"]
    leaf_id_nested = root["children"]["c"]["children"]["nested"]["id"]
    assert leaf_id_a == leaf_id_b == leaf_id_nested, (
        "Shared tensor should have one leaf id in the manifest"
    )

    loaded = serde.load(tmp_path)
    assert loaded["a"] is loaded["b"]
    assert loaded["a"] is loaded["c"]["nested"]
    assert torch.equal(loaded["a"], shared)


def test_shared_primitive_container_identity_is_preserved(tmp_path: pathlib.Path) -> None:
    shared = [1, 2, 3]
    original = {"a": shared, "b": shared}
    serde.save(original, tmp_path)

    root = _manifest(tmp_path)["root"]
    assert root["_t"] == "container"
    assert root["children"]["a"]["kind"] == "msgpack"
    assert root["children"]["a"]["id"] == root["children"]["b"]["id"]

    loaded = serde.load(tmp_path)
    assert loaded["a"] is loaded["b"]
    assert loaded["a"] == [1, 2, 3]


def test_shared_recursive_container_uses_ref_and_preserves_identity(tmp_path: pathlib.Path) -> None:
    shared = [torch.ones(2)]
    original = {"a": shared, "b": shared}
    serde.save(original, tmp_path)

    root = _manifest(tmp_path)["root"]
    first = root["children"]["a"]
    second = root["children"]["b"]
    assert first["_t"] == "container"
    assert second == {"_t": "ref", "target": first["node_id"]}

    loaded = serde.load(tmp_path)
    assert loaded["a"] is loaded["b"]
    assert loaded["a"][0] is loaded["b"][0]
    assert torch.equal(loaded["a"][0], shared[0])


def test_list_cycle_roundtrip(tmp_path: pathlib.Path) -> None:
    original: list[object] = []
    original.append(original)
    serde.save(original, tmp_path)

    root = _manifest(tmp_path)["root"]
    assert root["_t"] == "container"
    assert root["children"] == [{"_t": "ref", "target": root["node_id"]}]

    loaded = serde.load(tmp_path)
    assert loaded[0] is loaded


def test_dict_cycle_roundtrip(tmp_path: pathlib.Path) -> None:
    original: dict[str, object] = {}
    original["self"] = original
    serde.save(original, tmp_path)

    root = _manifest(tmp_path)["root"]
    assert root["_t"] == "container"
    assert root["children"]["self"] == {"_t": "ref", "target": root["node_id"]}

    loaded = serde.load(tmp_path)
    assert loaded["self"] is loaded


def test_nn_module_via_directory_serializer(tmp_path: pathlib.Path) -> None:
    """Directory escape hatch: serializers can own a subdir for heterogeneous layouts."""
    model = torch.nn.Linear(4, 2)
    serde.save(model, tmp_path)

    manifest = _manifest(tmp_path)
    assert manifest["root"]["_t"] == "dir"
    subdir = manifest["root"]["subdir"]
    assert (tmp_path / "dirs" / subdir / "model.pt").exists()

    loaded = serde.load(tmp_path)
    assert isinstance(loaded, torch.nn.Linear)
    x = torch.randn(1, 4)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))


def test_single_tensor_and_primitive_list_together(tmp_path: pathlib.Path) -> None:
    """List containing a tensor, an int, and a nested dict of tensors."""
    original = [torch.ones(2), 42, {"w": torch.zeros(3)}]
    serde.save(original, tmp_path)
    loaded = serde.load(tmp_path)
    assert torch.equal(loaded[0], original[0])
    assert loaded[1] == 42
    assert torch.equal(loaded[2]["w"], original[2]["w"])


def test_tensors_pt_contains_one_entry_per_unique_tensor(
    tmp_path: pathlib.Path,
) -> None:
    """Sanity: the batched blob holds exactly one entry per unique tensor."""
    t1 = torch.ones(3)
    t2 = torch.zeros(5)
    original = [t1, t2, t1]  # t1 appears twice — memo should dedup
    serde.save(original, tmp_path)

    bundle = torch.load(
        tmp_path / "leaves" / "torch_tensor" / "tensors.pt",
        weights_only=True,
        map_location="cpu",
    )
    assert len(bundle) == 2, "Shared tensor should appear once in the batched blob"


# ---------------------------------------------------------------------------
# Msgpack-subtree collapsing
# ---------------------------------------------------------------------------


def test_primitive_list_collapses_to_single_leaf(tmp_path: pathlib.Path) -> None:
    """``[1, 2, 3]`` should be one msgpack leaf, not a Container of three."""
    serde.save([1, 2, 3], tmp_path)

    root = _manifest(tmp_path)["root"]
    assert root["_t"] == "leaf"
    assert root["kind"] == "msgpack"

    assert serde.load(tmp_path) == [1, 2, 3]


def test_all_primitive_dict_collapses_to_single_leaf(tmp_path: pathlib.Path) -> None:
    """A whole config dict of primitives becomes one msgpack leaf."""
    original = {"lr": 0.001, "batch_size": 32, "name": "adam", "layers": [64, 32, 16]}
    serde.save(original, tmp_path)

    root = _manifest(tmp_path)["root"]
    assert root["_t"] == "leaf"
    assert root["kind"] == "msgpack"

    assert serde.load(tmp_path) == original


def test_msgpack_subtree_within_mixed_structure_collapses(tmp_path: pathlib.Path) -> None:
    """A primitive subtree inside a mixed dict collapses while the tensor stays separate.

    ``{"config": {primitives}, "weights": tensor}`` should produce:
      - root = Container (mixed — has a tensor somewhere)
      - root.children["config"] = Leaf (msgpack — whole subtree is primitives)
      - root.children["weights"] = Leaf (torch_tensor)
    """
    original = {
        "config": {"lr": 0.001, "batch_size": 32, "name": "adam"},
        "weights": torch.ones(4),
    }
    serde.save(original, tmp_path)

    manifest = _manifest(tmp_path)
    root = manifest["root"]
    assert root["_t"] == "container"
    config_node = root["children"]["config"]
    weights_node = root["children"]["weights"]
    assert config_node["_t"] == "leaf"
    assert config_node["kind"] == "msgpack"
    assert weights_node["_t"] == "leaf"
    assert weights_node["kind"] == "torch_tensor"

    loaded = serde.load(tmp_path)
    assert loaded["config"] == original["config"]
    assert torch.equal(loaded["weights"], original["weights"])


def test_deeply_nested_primitive_tree_is_one_leaf(tmp_path: pathlib.Path) -> None:
    """Arbitrary nesting of primitives shouldn't explode the manifest."""
    original = {"a": {"b": {"c": {"d": [1, {"e": "f"}]}}}}
    serde.save(original, tmp_path)

    root = _manifest(tmp_path)["root"]
    assert root["_t"] == "leaf", "Entire nested-primitive tree should be one leaf"
    assert serde.load(tmp_path) == original


def test_sibling_primitives_each_get_a_leaf_when_tensor_in_tree(
    tmp_path: pathlib.Path,
) -> None:
    """Corner of the design: primitives *siblings of a tensor* can't coalesce.

    They live at separate positions in the parent container, so each
    becomes its own msgpack leaf (sharing the same on-disk blob).
    This is the known limitation from the design discussion.
    """
    original = {"lr": 0.001, "step": 100, "name": "adam", "w": torch.ones(2)}
    serde.save(original, tmp_path)

    root = _manifest(tmp_path)["root"]
    assert root["_t"] == "container"
    # Three primitive siblings → three msgpack leaves (sharing one blob).
    msgpack_leaves = [c for c in root["children"].values() if c["_t"] == "leaf" and c["kind"] == "msgpack"]
    assert len(msgpack_leaves) == 3


def test_empty_containers_collapse_to_single_leaf(tmp_path: pathlib.Path) -> None:
    """Empty dict/list/tuple should be one msgpack leaf, not a Container."""
    for value in ({}, [], ()):
        d = tmp_path / f"v_{type(value).__name__}"
        d.mkdir()
        serde.save(value, d)
        root = _manifest(d)["root"]
        assert root["_t"] == "leaf", f"Empty {type(value).__name__} should collapse"
        loaded = serde.load(d)
        assert loaded == value
        assert type(loaded) is type(value)


# ---------------------------------------------------------------------------
# Dispatch plumbing: fast-path MRO walk + volatile-type content sensitivity
# ---------------------------------------------------------------------------


def test_nn_linear_dispatches_via_mro_walk(tmp_path: pathlib.Path) -> None:
    """``nn.Linear`` should resolve to ``TorchModuleSerializer`` via its MRO.

    Registry's by-type map only names ``torch.nn.modules.module.Module``
    — every Module subclass must find it through ``type.__mro__``.
    """
    serde.save(torch.nn.Linear(3, 2), tmp_path)
    root = _manifest(tmp_path)["root"]
    assert root["serializer"].endswith(".TorchModuleSerializer")


@pytest.mark.parametrize(
    ("tensor_value", "primitive_value", "expected_container_ser"),
    [
        ({"w": torch.ones(2)}, {"a": 1, "b": 2}, "DictSerializer"),
        ([torch.ones(2)], [1, 2, 3], "ListSerializer"),
        ((torch.ones(2),), (1, 2, 3), "TupleSerializer"),
    ],
    ids=["dict", "list", "tuple"],
)
def test_container_dispatch_is_content_sensitive(
    tmp_path: pathlib.Path,
    tensor_value: object,
    primitive_value: object,
    expected_container_ser: str,
) -> None:
    """Same concrete type must dispatch by content — verifies volatile-type handling.

    Without volatile-type treatment, the first save would pin
    ``type(obj)`` (dict/list/tuple) in the cache to whichever
    serializer won, and every subsequent save of that type would
    wrongly get the same serializer.
    """
    # First: the primitive form (should go to msgpack leaf).
    d_prim = tmp_path / "primitive"
    d_prim.mkdir()
    serde.save(primitive_value, d_prim)
    assert _manifest(d_prim)["root"]["serializer"].endswith(".MsgpackLeafSerializer")

    # Then: the tensor-containing form (should go to the container serializer,
    # bypassing any cache that might have pinned the type).
    d_tens = tmp_path / "tensor"
    d_tens.mkdir()
    serde.save(tensor_value, d_tens)
    assert _manifest(d_tens)["root"]["serializer"].endswith(f".{expected_container_ser}")

    # And back to primitive — still msgpack, not pinned to the container ser.
    d_prim2 = tmp_path / "primitive2"
    d_prim2.mkdir()
    serde.save(primitive_value, d_prim2)
    assert _manifest(d_prim2)["root"]["serializer"].endswith(".MsgpackLeafSerializer")


def test_repeated_tensor_saves_cache_dispatch(tmp_path: pathlib.Path) -> None:
    """Non-volatile types should hit the cache on repeat saves.

    We can't directly observe the cache, but we can spot-check
    correctness: dispatching torch.Tensor 100 times must yield the
    same serializer every call.  (If the cache were wrong, round-trip
    would fail or dispatch would drift to a different serializer.)
    """
    for i in range(5):
        d = tmp_path / f"t{i}"
        d.mkdir()
        t = torch.arange(i + 1, dtype=torch.float32)
        serde.save(t, d)
        root = _manifest(d)["root"]
        assert root["serializer"].endswith(".TorchTensorSerializer")
        assert torch.equal(serde.load(d), t)
