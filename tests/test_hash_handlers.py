import dataclasses
import importlib.util
import pathlib
import re
import types
from collections import ChainMap, Counter, defaultdict, deque
from fractions import Fraction
from typing import Any

import pytest
from misen_hash import stable_hash
from misen_hash.handler_base import CollectionHandler
from misen_hash.handlers.fallback import DillHandler
from misen_hash.handlers.optional import optional_handlers
from misen_hash.hash import incremental_hash


class _Custom:
    pass


@dataclasses.dataclass
class _DataWithList:
    values: list[int]


def test_dill_handler_matches_any_object() -> None:
    assert DillHandler.match(object()) is True


def test_incremental_hash_matches_hashing_dill_bytes() -> None:
    import dill
    from xxhash import xxh3_64_intdigest

    obj = {"a": [1, 2, 3], "b": {"x": 4}}
    incremental_digest = incremental_hash(lambda sink: dill.dump(obj, sink))
    buffered_digest = xxh3_64_intdigest(dill.dumps(obj), seed=0)
    assert incremental_digest == buffered_digest


def test_incremental_hash_hashes_incremental_chunks() -> None:
    from xxhash import xxh3_64_intdigest

    chunks: list[bytes | bytearray | memoryview] = [
        b"abc",
        bytearray(b"def"),
        memoryview(b"ghi"),
    ]

    def _serialize(sink: Any) -> None:
        for chunk in chunks:
            sink.write(chunk)

    incremental_digest = incremental_hash(_serialize)
    buffered_digest = xxh3_64_intdigest(b"abcdefghi", seed=0)
    assert incremental_digest == buffered_digest


def test_torch_module_handler_uses_collection_contract() -> None:
    pytest.importorskip("torch")
    from misen_hash.handlers.optional.torch import TorchModuleHandler

    assert issubclass(TorchModuleHandler, CollectionHandler)


def test_torch_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("torch") is None:
        assert all(handler.__name__ != "TorchModuleHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.torch import TorchModuleHandler

        assert TorchModuleHandler in optional_handlers


def test_stable_hash_falls_back_to_dill_handler() -> None:
    assert isinstance(stable_hash(_Custom()), int)


def test_dict_hash_is_order_independent_with_nested_unhashables() -> None:
    left = {"a": [1, 2], "b": {"c": 3}}
    right = {"b": {"c": 3}, "a": [1, 2]}
    assert stable_hash(left) == stable_hash(right)


def test_dataclass_hash_supports_unhashable_fields() -> None:
    left = _DataWithList(values=[1, 2, 3])
    right = _DataWithList(values=[1, 2, 3])
    assert stable_hash(left) == stable_hash(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (complex(1.5, -2.0), complex(1.5, -2.0)),
        (Fraction(3, 7), Fraction(3, 7)),
        (range(1, 10, 3), range(1, 10, 3)),
        (slice(1, 20, 2), slice(1, 20, 2)),
        (pathlib.PurePosixPath("a") / "b" / "c.txt", pathlib.PurePosixPath("a/b/c.txt")),
        (pathlib.PureWindowsPath(r"C:\a\b\c.txt"), pathlib.PureWindowsPath("C:/a/b/c.txt")),
        (memoryview(b"abc"), memoryview(b"abc")),
        (re.compile(r"ab+c", flags=re.IGNORECASE), re.compile(r"ab+c", flags=re.IGNORECASE)),
        (types.SimpleNamespace(a=1, b=[2, 3]), types.SimpleNamespace(b=[2, 3], a=1)),
        (deque([1, 2, 3]), deque([1, 2, 3])),
        (Counter({"a": 2, "b": 1}), Counter({"b": 1, "a": 2})),
        (
            defaultdict(list, {"a": [1, 2], "b": [3]}),
            defaultdict(list, {"b": [3], "a": [1, 2]}),
        ),
        (
            ChainMap({"a": [1, 2]}, {"b": 3}),
            ChainMap({"a": [1, 2]}, {"b": 3}),
        ),
        (
            types.MappingProxyType({"a": [1, 2], "b": {"c": 4}}),
            types.MappingProxyType({"b": {"c": 4}, "a": [1, 2]}),
        ),
    ],
)
def test_popular_stdlib_types_have_stable_hashes(left: object, right: object) -> None:
    assert stable_hash(left) == stable_hash(right)


def test_defaultdict_hash_includes_default_factory() -> None:
    left = defaultdict(list, {"a": 1})
    right = defaultdict(int, {"a": 1})
    assert stable_hash(left) != stable_hash(right)


def test_msgspec_struct_hash_supports_nested_unhashables() -> None:
    import msgspec

    class Point(msgspec.Struct):
        x: int
        y: list[int]

    left = Point(1, [2, 3])
    right = Point(1, [2, 3])
    assert stable_hash(left) == stable_hash(right)


def test_numpy_handlers_cover_arrays_and_scalars() -> None:
    np = pytest.importorskip("numpy")

    numeric_array = np.array([[1, 2], [3, 4]], dtype=np.int64)
    assert stable_hash(numeric_array) == stable_hash(np.array([[1, 2], [3, 4]], dtype=np.int64))
    assert stable_hash(numeric_array) != stable_hash(np.array([[1, 2], [3, 4]], dtype=np.int32))

    object_array = np.array([{"a": [1, 2]}, [3, 4]], dtype=object)
    assert stable_hash(object_array) == stable_hash(np.array([{"a": [1, 2]}, [3, 4]], dtype=object))
    assert stable_hash(np.int64(7)) != stable_hash(np.int32(7))


def test_pandas_handlers_cover_dataframe_series_index() -> None:
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}, index=[10, 20])
    assert stable_hash(df) == stable_hash(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}, index=[10, 20]))
    assert stable_hash(df) != stable_hash(df[["b", "a"]])

    series = pd.Series([1, 2, 3], index=["x", "y", "z"], name="vals")
    assert stable_hash(series) == stable_hash(pd.Series([1, 2, 3], index=["x", "y", "z"], name="vals"))

    index = pd.Index(["a", "b", "c"], name="idx")
    assert stable_hash(index) == stable_hash(pd.Index(["a", "b", "c"], name="idx"))

    fallback_df = pd.DataFrame({"a": [{"k": [1, 2]}, {"k": [3]}]})
    assert stable_hash(fallback_df) == stable_hash(pd.DataFrame({"a": [{"k": [1, 2]}, {"k": [3]}]}))


def test_pydantic_model_handler() -> None:
    pydantic = pytest.importorskip("pydantic")

    class Model(pydantic.BaseModel):
        x: int
        y: list[int]

    left = Model(x=1, y=[2, 3])
    right = Model(x=1, y=[2, 3])
    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(Model(x=1, y=[2]))


def test_attrs_handler() -> None:
    attrs = pytest.importorskip("attrs")

    @attrs.define
    class Config:
        x: int
        values: list[int]

    left = Config(x=1, values=[2, 3])
    right = Config(x=1, values=[2, 3])
    assert stable_hash(left) == stable_hash(right)


def test_polars_handlers() -> None:
    pl = pytest.importorskip("polars")

    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert stable_hash(df) == stable_hash(pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    assert stable_hash(df) != stable_hash(pl.DataFrame({"a": [2, 1], "b": ["x", "y"]}))

    series = pl.Series("a", [1, 2, 3])
    assert stable_hash(series) == stable_hash(pl.Series("a", [1, 2, 3]))

    lazy = df.lazy().select((pl.col("a") + 1).alias("a_plus_one"))
    assert stable_hash(lazy) == stable_hash(df.lazy().select((pl.col("a") + 1).alias("a_plus_one")))


def test_pyarrow_handlers() -> None:
    pa = pytest.importorskip("pyarrow")

    array = pa.array([1, 2, 3])
    assert stable_hash(array) == stable_hash(pa.array([1, 2, 3]))

    table = pa.table({"a": [1, 2], "b": ["x", "y"]})
    assert stable_hash(table) == stable_hash(pa.table({"a": [1, 2], "b": ["x", "y"]}))

    schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    assert stable_hash(schema) == stable_hash(pa.schema([("a", pa.int64()), ("b", pa.string())]))


def test_xarray_handlers() -> None:
    xr = pytest.importorskip("xarray")

    array = xr.DataArray(
        [[1, 2], [3, 4]],
        dims=["row", "col"],
        coords={"row": [10, 20], "col": ["x", "y"]},
        name="values",
    )
    assert stable_hash(array) == stable_hash(
        xr.DataArray(
            [[1, 2], [3, 4]],
            dims=["row", "col"],
            coords={"row": [10, 20], "col": ["x", "y"]},
            name="values",
        )
    )

    dataset = xr.Dataset({"values": array})
    assert stable_hash(dataset) == stable_hash(xr.Dataset({"values": array}))


def test_scipy_sparse_handler() -> None:
    sparse = pytest.importorskip("scipy.sparse")

    matrix = sparse.csr_matrix([[0, 1], [2, 0]])
    assert stable_hash(matrix) == stable_hash(sparse.csr_matrix([[0, 1], [2, 0]]))
    assert stable_hash(matrix) != stable_hash(sparse.csr_matrix([[0, 1], [0, 2]]))


def test_pillow_handler() -> None:
    image_mod = pytest.importorskip("PIL.Image")

    left = image_mod.new("RGB", (2, 2), color=(10, 20, 30))
    right = image_mod.new("RGB", (2, 2), color=(10, 20, 30))
    different = image_mod.new("RGB", (2, 2), color=(11, 20, 30))
    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_networkx_handler() -> None:
    nx = pytest.importorskip("networkx")

    first = nx.MultiDiGraph()
    first.graph["meta"] = {"version": 1}
    first.add_node("a", payload=[1, 2])
    first.add_node("b", payload=[3])
    first.add_edge("a", "b", key="edge-1", weight={"x": 5})

    second = nx.MultiDiGraph()
    second.graph["meta"] = {"version": 1}
    second.add_node("b", payload=[3])
    second.add_node("a", payload=[1, 2])
    second.add_edge("a", "b", key="edge-1", weight={"x": 5})

    assert stable_hash(first) == stable_hash(second)


def test_sklearn_handler() -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")

    left = linear_model.LogisticRegression(C=1.0, max_iter=50, fit_intercept=True)
    right = linear_model.LogisticRegression(C=1.0, max_iter=50, fit_intercept=True)
    different = linear_model.LogisticRegression(C=0.9, max_iter=50, fit_intercept=True)

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)
