"""Round-trip and batching tests for library serializers in :mod:`serde`.

Focused on what's importable in the test env (numpy, sympy, msgspec,
torch) — the spike for the design proof.  Other libs (pandas, jax,
keras, ...) are guarded by :func:`pytest.importorskip` so they run in
CI environments that include them.
"""
# ruff: noqa: D103, S101

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from misen.utils import serde

if TYPE_CHECKING:
    import pathlib

numpy = pytest.importorskip("numpy")


def _manifest(directory: pathlib.Path) -> dict:
    return json.loads((directory / serde.MANIFEST_FILENAME).read_text())


# ---------------------------------------------------------------------------
# numpy — the second batching leaf lib after torch
# ---------------------------------------------------------------------------


def test_numpy_array_roundtrip(tmp_path: pathlib.Path) -> None:
    arr = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
    serde.save(arr, tmp_path)
    loaded = serde.load(tmp_path)
    assert isinstance(loaded, numpy.ndarray)
    assert loaded.dtype == arr.dtype
    assert numpy.array_equal(loaded, arr)


def test_numpy_nested_dict_batches_into_single_npz(tmp_path: pathlib.Path) -> None:
    """Nested ndarrays pack into ONE ``arrays.npz`` — the batching win."""
    original = {
        "encoder": {"w": numpy.ones(2), "b": numpy.zeros(3)},
        "decoder": {"w": numpy.ones(5), "b": numpy.zeros(1)},
    }
    serde.save(original, tmp_path)

    arrays_blob = tmp_path / "leaves" / "ndarray" / "arrays.npz"
    assert arrays_blob.exists(), "All ndarrays should land in a single batched npz"
    # 4 arrays in one archive.
    with numpy.load(arrays_blob) as npz:
        assert len(npz.files) == 4

    loaded = serde.load(tmp_path)
    for section in ("encoder", "decoder"):
        for k in ("w", "b"):
            assert numpy.array_equal(loaded[section][k], original[section][k])


def test_numpy_shared_array_identity_preserved(tmp_path: pathlib.Path) -> None:
    shared = numpy.ones(1000)
    original = {"a": shared, "b": shared, "c": {"nested": shared}}
    serde.save(original, tmp_path)

    # Single entry in the batched npz for the shared array.
    with numpy.load(tmp_path / "leaves" / "ndarray" / "arrays.npz") as npz:
        assert len(npz.files) == 1

    loaded = serde.load(tmp_path)
    assert loaded["a"] is loaded["b"]
    assert loaded["a"] is loaded["c"]["nested"]
    assert numpy.array_equal(loaded["a"], shared)


def test_numpy_scalar_batched_into_single_msgpack(tmp_path: pathlib.Path) -> None:
    """NumpyScalarSerializer bundles scalars in a shared msgpack blob."""
    original = {"a": numpy.float32(1.5), "b": numpy.int64(42), "c": numpy.bool_(True)}
    serde.save(original, tmp_path)

    scalars_blob = tmp_path / "leaves" / "numpy_scalar" / "scalars.msgpack"
    assert scalars_blob.exists()

    loaded = serde.load(tmp_path)
    assert loaded["a"] == pytest.approx(1.5)
    assert type(loaded["a"]) is numpy.float32
    assert loaded["b"] == 42
    assert type(loaded["b"]) is numpy.int64
    assert loaded["c"] is numpy.bool_(True)


def test_numpy_masked_array_roundtrip_via_directory(tmp_path: pathlib.Path) -> None:
    data = numpy.arange(6, dtype=numpy.float64).reshape(2, 3)
    mask = numpy.array([[True, False, False], [False, True, False]])
    original = numpy.ma.MaskedArray(data, mask=mask, fill_value=-1.0)
    serde.save(original, tmp_path)

    root = _manifest(tmp_path)["root"]
    assert root["_t"] == "dir"
    assert (tmp_path / "dirs" / root["subdir"] / "data.npy").exists()
    assert (tmp_path / "dirs" / root["subdir"] / "mask.npy").exists()

    loaded = serde.load(tmp_path)
    assert isinstance(loaded, numpy.ma.MaskedArray)
    assert numpy.array_equal(loaded.data, original.data)
    assert numpy.array_equal(loaded.mask, original.mask)
    assert loaded.fill_value == original.fill_value


# ---------------------------------------------------------------------------
# Cross-lib batching: tensors + ndarrays in one structure produce two blobs
# ---------------------------------------------------------------------------


def test_torch_and_numpy_coexist(tmp_path: pathlib.Path) -> None:
    """A dict mixing torch Tensors and numpy arrays produces two batched blobs."""
    torch = pytest.importorskip("torch")

    original = {
        "torch_weights": {"w": torch.ones(3), "b": torch.zeros(2)},
        "numpy_stats": {"mean": numpy.ones(5), "std": numpy.zeros(5)},
        "hyperparams": {"lr": 0.001, "epochs": 10},  # collapses to one msgpack leaf
    }
    serde.save(original, tmp_path)

    assert (tmp_path / "leaves" / "torch_tensor" / "tensors.pt").exists()
    assert (tmp_path / "leaves" / "ndarray" / "arrays.npz").exists()
    assert (tmp_path / "leaves" / "msgpack" / "data.msgpack").exists()

    loaded = serde.load(tmp_path)
    assert torch.equal(loaded["torch_weights"]["w"], original["torch_weights"]["w"])
    assert numpy.array_equal(loaded["numpy_stats"]["mean"], original["numpy_stats"]["mean"])
    assert loaded["hyperparams"] == original["hyperparams"]


# ---------------------------------------------------------------------------
# Directory serializers for other libs (spot-check the port pattern)
# ---------------------------------------------------------------------------


def test_sympy_expression_roundtrip(tmp_path: pathlib.Path) -> None:
    sympy = pytest.importorskip("sympy")
    x, y = sympy.symbols("x y")
    expr = x**2 + 2 * y + 1
    serde.save(expr, tmp_path)

    root = _manifest(tmp_path)["root"]
    assert root["_t"] == "dir"
    assert root["serializer"].endswith(".SympyExprSerializer")

    loaded = serde.load(tmp_path)
    assert loaded == expr


# ---------------------------------------------------------------------------
# Structured-data libs — recursive field walk (attrs / pydantic / msgspec / dataclass)
# ---------------------------------------------------------------------------
#
# These four are now Container-style serializers: each field dispatches
# through :func:`ctx.encode`, so a field holding a tensor / ndarray /
# DataFrame round-trips through its own specialized serializer.  Nested
# struct-within-struct with array fields also works.
#
# Classes are defined at module scope so the per-instance meta
# (``*_type: "tests.test_serde_libs._MsgspecConfig"``) can re-import
# them on decode.

import dataclasses as _dataclasses  # noqa: E402 — module-level imports for test fixtures
import enum  # noqa: E402
from typing import NamedTuple  # noqa: E402

import msgspec as _msgspec_top_level  # noqa: E402

_attrs = pytest.importorskip("attrs")
_pydantic = pytest.importorskip("pydantic")


class _MsgspecConfig(_msgspec_top_level.Struct):
    """Basic Struct — round-trips through the recursive Container path."""

    lr: float
    name: str
    layers: list[int]


class _MsgspecInner(_msgspec_top_level.Struct):
    label: str
    arr: Any  # ndarray


class _MsgspecOuter(_msgspec_top_level.Struct):
    name: str
    inner: _MsgspecInner


@_attrs.define
class _AttrsConfig:
    name: str
    step: int
    arr: Any  # ndarray


class _PydanticState(_pydantic.BaseModel):
    model_config = _pydantic.ConfigDict(arbitrary_types_allowed=True)
    name: str
    step: int
    weights: Any  # torch.Tensor


class _PydanticNested(_pydantic.BaseModel):
    model_config = _pydantic.ConfigDict(arbitrary_types_allowed=True)
    label: str
    inner: _PydanticState


@_dataclasses.dataclass
class _DataclassConfig:
    name: str
    step: int
    arr: Any  # ndarray


@_dataclasses.dataclass
class _DataclassInner:
    label: str
    weights: Any  # torch.Tensor


@_dataclasses.dataclass
class _DataclassOuter:
    name: str
    inner: _DataclassInner


@_dataclasses.dataclass(frozen=True)
class _DataclassFrozenPostInit:
    x: int
    doubled: int = _dataclasses.field(init=False, default=0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "doubled", self.x * 2)


class _NestingOuter:
    """Holds nested classes for each structured-data serializer.

    Nested classes have a ``__qualname__`` like ``_NestingOuter.Inner``
    (containing a dot), which used to break decode because the naive
    ``rpartition('.')`` split put the class name in the module path.
    """

    @_dataclasses.dataclass
    class Dataclass:
        label: str

    @_attrs.define
    class Attrs:
        label: str

    class Msgspec(_msgspec_top_level.Struct):
        label: str

    class Pydantic(_pydantic.BaseModel):
        label: str

    class Named(NamedTuple):
        label: str
        count: int

    class Color(enum.Enum):
        RED = 1


def test_msgspec_struct_roundtrip(tmp_path: pathlib.Path) -> None:
    """Recursive field walk — each field dispatches independently."""
    original = _MsgspecConfig(lr=0.001, name="adam", layers=[64, 32, 16])
    serde.save(original, tmp_path)

    root = _manifest(tmp_path)["root"]
    assert root["_t"] == "container"  # Container now, not DirectoryLeaf
    assert root["serializer"].endswith(".MsgspecStructSerializer")

    loaded = serde.load(tmp_path)
    assert loaded == original
    assert type(loaded) is _MsgspecConfig


def test_msgspec_struct_with_array_field_roundtrips(tmp_path: pathlib.Path) -> None:
    """v1 gap closed: msgspec.msgpack can't encode ndarrays, but ctx.encode can."""
    original = _MsgspecOuter(name="outer", inner=_MsgspecInner(label="x", arr=numpy.arange(5)))
    serde.save(original, tmp_path)

    # The ndarray is in the numpy leaf batch, not inside the msgpack blob.
    assert (tmp_path / "leaves" / "ndarray" / "arrays.npz").exists()

    loaded = serde.load(tmp_path)
    assert type(loaded) is _MsgspecOuter
    assert type(loaded.inner) is _MsgspecInner
    assert loaded.name == "outer"
    assert loaded.inner.label == "x"
    assert numpy.array_equal(loaded.inner.arr, original.inner.arr)


def test_attrs_basic_roundtrip(tmp_path: pathlib.Path) -> None:
    original = _AttrsConfig(name="adam", step=100, arr=numpy.array([1, 2, 3]))
    serde.save(original, tmp_path)

    root = _manifest(tmp_path)["root"]
    assert root["_t"] == "container"
    assert root["serializer"].endswith(".AttrsSerializer")

    loaded = serde.load(tmp_path)
    assert type(loaded) is _AttrsConfig
    assert loaded.name == "adam"
    assert loaded.step == 100
    assert numpy.array_equal(loaded.arr, original.arr)


def test_attrs_with_ndarray_lands_in_numpy_batch(tmp_path: pathlib.Path) -> None:
    """attrs field holding an ndarray routes to the numpy leaf blob."""
    original = _AttrsConfig(name="test", step=7, arr=numpy.arange(20))
    serde.save(original, tmp_path)

    assert (tmp_path / "leaves" / "ndarray" / "arrays.npz").exists()
    loaded = serde.load(tmp_path)
    assert numpy.array_equal(loaded.arr, original.arr)


def test_pydantic_basic_roundtrip(tmp_path: pathlib.Path) -> None:
    torch = pytest.importorskip("torch")
    original = _PydanticState(name="adam", step=100, weights=torch.ones(4))
    serde.save(original, tmp_path)

    root = _manifest(tmp_path)["root"]
    assert root["_t"] == "container"
    assert root["serializer"].endswith(".PydanticModelSerializer")

    loaded = serde.load(tmp_path)
    assert type(loaded) is _PydanticState
    assert loaded.name == "adam"
    assert loaded.step == 100
    assert torch.equal(loaded.weights, original.weights)


def test_pydantic_model_with_tensor_field(tmp_path: pathlib.Path) -> None:
    """v1 gap closed: model_dump_json can't encode tensors, but ctx.encode can."""
    torch = pytest.importorskip("torch")
    original = _PydanticState(name="adam", step=50, weights=torch.arange(6, dtype=torch.float32))
    serde.save(original, tmp_path)

    # Tensor goes to torch leaf batch; scalar fields collapse to msgpack.
    assert (tmp_path / "leaves" / "torch_tensor" / "tensors.pt").exists()

    loaded = serde.load(tmp_path)
    assert torch.equal(loaded.weights, original.weights)


def test_pydantic_nested_models_with_tensor(tmp_path: pathlib.Path) -> None:
    """Nested BaseModel-inside-BaseModel with a tensor deep inside."""
    torch = pytest.importorskip("torch")
    inner = _PydanticState(name="inner", step=1, weights=torch.ones(3))
    outer = _PydanticNested(label="top", inner=inner)
    serde.save(outer, tmp_path)

    loaded = serde.load(tmp_path)
    assert type(loaded) is _PydanticNested
    assert type(loaded.inner) is _PydanticState
    assert loaded.label == "top"
    assert loaded.inner.name == "inner"
    assert torch.equal(loaded.inner.weights, inner.weights)


def test_dataclass_basic_roundtrip(tmp_path: pathlib.Path) -> None:
    """Recursive field walk — each field dispatches independently."""
    original = _DataclassConfig(name="adam", step=100, arr=numpy.array([1, 2, 3]))
    serde.save(original, tmp_path)

    root = _manifest(tmp_path)["root"]
    assert root["_t"] == "container"
    assert root["serializer"].endswith(".DataclassSerializer")

    loaded = serde.load(tmp_path)
    assert type(loaded) is _DataclassConfig
    assert loaded.name == "adam"
    assert loaded.step == 100
    assert numpy.array_equal(loaded.arr, original.arr)


def test_dataclass_with_ndarray_lands_in_numpy_batch(tmp_path: pathlib.Path) -> None:
    """v1 gap closed: tagged msgpack can't encode ndarrays, but ctx.encode can."""
    original = _DataclassConfig(name="test", step=7, arr=numpy.arange(20))
    serde.save(original, tmp_path)

    # The ndarray is in the numpy leaf batch, not inside the msgpack blob.
    assert (tmp_path / "leaves" / "ndarray" / "arrays.npz").exists()
    loaded = serde.load(tmp_path)
    assert numpy.array_equal(loaded.arr, original.arr)


def test_dataclass_nested_with_tensor(tmp_path: pathlib.Path) -> None:
    """Nested dataclass-inside-dataclass with a tensor deep inside."""
    torch = pytest.importorskip("torch")
    inner = _DataclassInner(label="x", weights=torch.ones(3))
    outer = _DataclassOuter(name="outer", inner=inner)
    serde.save(outer, tmp_path)

    # Tensor lands in the torch leaf batch, not in the msgpack blob.
    assert (tmp_path / "leaves" / "torch_tensor" / "tensors.pt").exists()

    loaded = serde.load(tmp_path)
    assert type(loaded) is _DataclassOuter
    assert type(loaded.inner) is _DataclassInner
    assert loaded.name == "outer"
    assert loaded.inner.label == "x"
    assert torch.equal(loaded.inner.weights, inner.weights)


def test_dataclass_frozen_with_init_false_field(tmp_path: pathlib.Path) -> None:
    """``init=False`` fields restore via ``object.__setattr__`` (works for frozen too)."""
    original = _DataclassFrozenPostInit(x=7)
    assert original.doubled == 14
    serde.save(original, tmp_path)

    loaded = serde.load(tmp_path)
    assert type(loaded) is _DataclassFrozenPostInit
    assert loaded.x == 7
    assert loaded.doubled == 14


def test_dataclass_dispatch_stable_across_mixed_instances(tmp_path: pathlib.Path) -> None:
    """Same dataclass type with a native then a non-native instance both round-trip.

    Guards the caching bug that would surface if the MsgpackLeafSerializer
    claimed pure-native dataclasses: the first pure-native instance would
    cache ``Config → MsgpackLeafSerializer`` and a later ndarray-bearing
    instance of the same class would fail under the cached dispatch.
    """
    native = _DataclassConfig(name="native", step=1, arr=42)
    serde.save(native, tmp_path / "a")
    assert serde.load(tmp_path / "a") == native

    mixed = _DataclassConfig(name="mixed", step=2, arr=numpy.arange(4))
    serde.save(mixed, tmp_path / "b")
    loaded = serde.load(tmp_path / "b")
    assert loaded.name == "mixed"
    assert numpy.array_equal(loaded.arr, mixed.arr)


# ---------------------------------------------------------------------------
# Nested classes — shared ``import_by_qualified_name`` helper handles qualnames
# like ``Outer.Inner`` which used to break the naive ``rpartition('.')`` split.
# ---------------------------------------------------------------------------


def test_nested_dataclass_roundtrip(tmp_path: pathlib.Path) -> None:
    obj = _NestingOuter.Dataclass(label="inner")
    serde.save(obj, tmp_path)
    loaded = serde.load(tmp_path)
    assert type(loaded) is _NestingOuter.Dataclass
    assert loaded.label == "inner"


def test_nested_attrs_roundtrip(tmp_path: pathlib.Path) -> None:
    obj = _NestingOuter.Attrs(label="inner")
    serde.save(obj, tmp_path)
    loaded = serde.load(tmp_path)
    assert type(loaded) is _NestingOuter.Attrs
    assert loaded.label == "inner"


def test_nested_msgspec_roundtrip(tmp_path: pathlib.Path) -> None:
    obj = _NestingOuter.Msgspec(label="inner")
    serde.save(obj, tmp_path)
    loaded = serde.load(tmp_path)
    assert type(loaded) is _NestingOuter.Msgspec
    assert loaded.label == "inner"


def test_nested_pydantic_roundtrip(tmp_path: pathlib.Path) -> None:
    obj = _NestingOuter.Pydantic(label="inner")
    serde.save(obj, tmp_path)
    loaded = serde.load(tmp_path)
    assert type(loaded) is _NestingOuter.Pydantic
    assert loaded.label == "inner"


def test_nested_namedtuple_roundtrip(tmp_path: pathlib.Path) -> None:
    """Msgpack-tagged namedtuple with a dotted qualname also resolves now."""
    obj = _NestingOuter.Named(label="inner", count=3)
    serde.save(obj, tmp_path)
    loaded = serde.load(tmp_path)
    assert type(loaded) is _NestingOuter.Named
    assert loaded == obj


def test_nested_enum_roundtrip(tmp_path: pathlib.Path) -> None:
    """Msgpack-tagged enum with a dotted qualname also resolves now."""
    obj = _NestingOuter.Color.RED
    serde.save(obj, tmp_path)
    loaded = serde.load(tmp_path)
    assert loaded is _NestingOuter.Color.RED


# ---------------------------------------------------------------------------
# Import hygiene: all lib modules import cleanly even when deps are missing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name",
    [
        "altair",
        "attrs",
        "catboost",
        "dataclass",
        "faiss",
        "geopandas",
        "hf_datasets",
        "jax",
        "keras",
        "lightgbm",
        "msgspec_struct",
        "numpy",
        "onnx",
        "pandas",
        "pillow",
        "plotly",
        "polars",
        "pyarrow",
        "pydantic",
        "scipy",
        "shapely",
        "sklearn",
        "stdlib",
        "sympy",
        "tensorflow",
        "tokenizers",
        "torch",
        "transformers",
        "xarray",
        "xgboost",
    ],
)
def test_lib_module_imports_cleanly(module_name: str) -> None:
    """Every lib module must import without error, guard missing deps."""
    import importlib

    mod = importlib.import_module(f"misen.utils.serde.libs.{module_name}")
    # The ``<name>_serializers`` list attribute must exist and be a list.
    serializers_attr = f"{module_name}_serializers"
    assert hasattr(mod, serializers_attr), f"{module_name} missing {serializers_attr}"
    assert isinstance(getattr(mod, serializers_attr), list)
