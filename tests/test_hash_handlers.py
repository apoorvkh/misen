import dataclasses
import functools
import importlib.util
import inspect
import ipaddress
import pathlib
import re
import types
import typing
from collections import ChainMap, Counter, defaultdict, deque
from fractions import Fraction
from typing import ForwardRef, TypeVar

import pytest
from misen.utils.hashing import stable_hash
from misen.utils.hashing.handlers.optional import optional_handlers


class _Custom:
    pass


@dataclasses.dataclass
class _DataWithList:
    values: list[int]


def _optional_handler_names() -> set[str]:
    return {handler.__name__ for handler in optional_handlers}


@pytest.mark.parametrize(
    ("module_name", "handler_name"),
    [
        ("attrs", "AttrsHandler"),
        ("msgspec", "MsgspecStructHandler"),
        ("nltk", "NLTKCFGHandler"),
        ("numpy", "NumpyDTypeHandler"),
        ("numpy", "NumpyScalarHandler"),
        ("pyarrow", "PyArrowSchemaHandler"),
        ("pydantic", "PydanticModelHandler"),
        ("sympy", "SymPyBasicHandler"),
        ("transformers", "TransformersConfigHandler"),
    ],
)
def test_kept_optional_handlers_are_registered(module_name: str, handler_name: str) -> None:
    handler_names = _optional_handler_names()
    if importlib.util.find_spec(module_name) is None:
        assert handler_name not in handler_names
    else:
        assert handler_name in handler_names


@pytest.mark.parametrize(
    "handler_name",
    [
        "AltairChartHandler",
        "DaskCollectionHandler",
        "HFDatasetHandler",
        "HFDatasetDictHandler",
        "HFTokenizerHandler",
        "JaxArrayHandler",
        "KerasModelHandler",
        "LightGBMBoosterHandler",
        "LightGBMSklearnModelHandler",
        "MatplotlibAxesHandler",
        "MatplotlibFigureHandler",
        "NetworkXGraphHandler",
        "NLTKTreeHandler",
        "NumpyArrayHandler",
        "OpenCVDMatchHandler",
        "OpenCVKeyPointHandler",
        "OpenCVUMatHandler",
        "PandasDataFrameHandler",
        "PandasIndexHandler",
        "PandasSeriesHandler",
        "PillowImageHandler",
        "PlotlyFigureHandler",
        "PolarsDataFrameHandler",
        "PolarsLazyFrameHandler",
        "PolarsSeriesHandler",
        "PyArrowArrayHandler",
        "PyArrowChunkedArrayHandler",
        "PyArrowRecordBatchHandler",
        "PyArrowScalarHandler",
        "PyArrowTableHandler",
        "RustworkxGraphHandler",
        "SciPySparseHandler",
        "SeabornGridHandler",
        "SentencePieceProcessorHandler",
        "SkimageTransformHandler",
        "SklearnEstimatorHandler",
        "SpacyLanguageHandler",
        "StatsmodelsModelHandler",
        "StatsmodelsResultsHandler",
        "TensorFlowTensorHandler",
        "TensorFlowVariableHandler",
        "TorchModuleHandler",
        "TorchTensorHandler",
        "TransformersModelHandler",
        "TransformersTokenizerHandler",
        "XarrayDataArrayHandler",
        "XarrayDatasetHandler",
        "XGBoostBoosterHandler",
        "XGBoostSklearnModelHandler",
    ],
)
def test_removed_optional_handlers_are_not_registered(handler_name: str) -> None:
    assert handler_name not in _optional_handler_names()


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(_Custom(), id="custom-object"),
        pytest.param(lambda x: x, id="function"),
        pytest.param(len, id="builtin-function"),
        pytest.param("abc".upper, id="bound-method"),
        pytest.param(functools.partial(pow, 2, exp=5), id="partial"),
        pytest.param((lambda: None).__code__, id="code"),
        pytest.param(inspect.signature(lambda x: x), id="signature"),
        pytest.param(next(iter(inspect.signature(lambda x: x).parameters.values())), id="parameter"),
        pytest.param(typing.List[int], id="typing-alias"),
        pytest.param(TypeVar("T"), id="typevar"),
        pytest.param(ForwardRef("Demo"), id="forwardref"),
        pytest.param(int | str, id="union-type"),
    ],
)
def test_stable_hash_rejects_unsupported_runtime_types(value: object) -> None:
    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(value)


def test_stable_hash_supports_recursive_list() -> None:
    left: list[object] = []
    left.append(left)

    right: list[object] = []
    right.append(right)

    assert stable_hash(left) == stable_hash(right)


def test_stable_hash_supports_recursive_dict() -> None:
    left: dict[str, object] = {}
    left["self"] = left

    right: dict[str, object] = {}
    right["self"] = right

    assert stable_hash(left) == stable_hash(right)


def test_recursive_dict_hash_is_order_independent() -> None:
    left: dict[str, object] = {}
    left["a"] = 1
    left["self"] = left

    right: dict[str, object] = {}
    right["self"] = right
    right["a"] = 1

    assert stable_hash(left) == stable_hash(right)


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
        (ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address("127.0.0.1")),
        (ipaddress.ip_network("10.0.0.0/24"), ipaddress.ip_network("10.0.0.0/24")),
        (ipaddress.ip_interface("10.0.0.10/24"), ipaddress.ip_interface("10.0.0.10/24")),
        (re.compile(r"ab+c", flags=re.IGNORECASE), re.compile(r"ab+c", flags=re.IGNORECASE)),
        (types.SimpleNamespace(a=1, b=[2, 3]), types.SimpleNamespace(b=[2, 3], a=1)),
        (deque([1, 2, 3]), deque([1, 2, 3])),
        (Counter({"a": 2, "b": 1}), Counter({"b": 1, "a": 2})),
        (
            defaultdict(None, {"a": [1, 2], "b": [3]}),
            defaultdict(None, {"b": [3], "a": [1, 2]}),
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


def test_defaultdict_with_type_default_factory() -> None:
    left = defaultdict(list, {"a": 1})
    right = defaultdict(list, {"a": 1})
    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(defaultdict(int, {"a": 1}))


def test_dict_view_handlers_cover_keys_values_items() -> None:
    left = {"a": [1, 2], "b": {"x": 3}}
    right = {"b": {"x": 3}, "a": [1, 2]}

    assert stable_hash(left.keys()) == stable_hash(right.keys())
    assert stable_hash(left.values()) != stable_hash({"a": [1, 2], "b": {"x": 4}}.values())
    assert stable_hash(left.items()) == stable_hash(right.items())


def test_msgspec_struct_hash_supports_nested_unhashables() -> None:
    import msgspec

    class Point(msgspec.Struct):
        x: int
        y: list[int]

    left = Point(1, [2, 3])
    right = Point(1, [2, 3])
    assert stable_hash(left) == stable_hash(right)


def test_numpy_handlers_cover_dtypes_and_scalars() -> None:
    np = pytest.importorskip("numpy")

    assert stable_hash(np.dtype("int64")) == stable_hash(np.dtype("int64"))
    assert stable_hash(np.dtype("int64")) != stable_hash(np.dtype("int32"))
    assert stable_hash(np.int64(7)) == stable_hash(np.int64(7))
    assert stable_hash(np.int64(7)) != stable_hash(np.int32(7))


def test_numpy_array_is_unsupported() -> None:
    np = pytest.importorskip("numpy")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(np.array([[1, 2], [3, 4]], dtype=np.int64))


def test_jax_array_is_unsupported() -> None:
    jnp = pytest.importorskip("jax.numpy")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(jnp.array([[1, 2], [3, 4]], dtype=jnp.int32))


def test_tensorflow_tensor_is_unsupported() -> None:
    tf = pytest.importorskip("tensorflow")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(tf.constant([[1, 2], [3, 4]], dtype=tf.int32))


def test_tensorflow_keras_model_is_unsupported() -> None:
    tf = pytest.importorskip("tensorflow")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(4, activation="relu"),
        ]
    )

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(model)


def test_dask_collection_is_unsupported() -> None:
    dask_array = pytest.importorskip("dask.array")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(dask_array.arange(10, chunks=(5,)).reshape((2, 5)))


def test_hugging_face_dataset_is_unsupported() -> None:
    datasets = pytest.importorskip("datasets")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(datasets.Dataset.from_dict({"x": [1, 2], "y": ["a", "b"]}))


def test_transformers_config_handler() -> None:
    transformers = pytest.importorskip("transformers")

    left = transformers.BertConfig(hidden_size=16, num_attention_heads=2, intermediate_size=32)
    right = transformers.BertConfig(hidden_size=16, num_attention_heads=2, intermediate_size=32)
    different = transformers.BertConfig(hidden_size=32, num_attention_heads=2, intermediate_size=32)

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_transformers_model_is_unsupported() -> None:
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    model = transformers.BertModel(
        transformers.BertConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    )

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(model)


def test_tokenizers_handler_is_unsupported() -> None:
    tokenizers = pytest.importorskip("tokenizers")

    tokenizer = tokenizers.Tokenizer.from_str(
        """
        {"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,
         "pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,
         "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"world":2},"unk_token":"[UNK]"}}
        """
    )

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(tokenizer)


def test_sentencepiece_processor_is_unsupported() -> None:
    sentencepiece = pytest.importorskip("sentencepiece")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(sentencepiece.SentencePieceProcessor())


def test_sympy_handler() -> None:
    sympy = pytest.importorskip("sympy")

    x = sympy.Symbol("x")
    left = (x + 1) ** 2
    right = (x + 1) ** 2
    different = (x + 2) ** 2

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_nltk_cfg_handler() -> None:
    nltk = pytest.importorskip("nltk")

    left = nltk.CFG.fromstring("S -> NP VP\nNP -> 'I'\nVP -> 'run'")
    right = nltk.CFG.fromstring("S -> NP VP\nNP -> 'I'\nVP -> 'run'")
    different = nltk.CFG.fromstring("S -> NP VP\nNP -> 'I'\nVP -> 'walk'")

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_nltk_tree_is_unsupported() -> None:
    nltk = pytest.importorskip("nltk")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(nltk.Tree("S", [nltk.Tree("NP", ["I"]), nltk.Tree("VP", ["run"])]))


def test_plotly_figure_is_unsupported() -> None:
    graph_objects = pytest.importorskip("plotly.graph_objects")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(graph_objects.Figure(data=[graph_objects.Scatter(y=[1, 2, 3])]))


def test_altair_chart_is_unsupported() -> None:
    altair = pytest.importorskip("altair")
    pd = pytest.importorskip("pandas")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(altair.Chart(pd.DataFrame({"x": [0, 1], "y": [1, 3]})).mark_line().encode(x="x:Q", y="y:Q"))


def test_matplotlib_figure_is_unsupported() -> None:
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg", force=True)
    pyplot = pytest.importorskip("matplotlib.pyplot")
    figure, _ax = pyplot.subplots()
    try:
        with pytest.raises(TypeError, match="explicit handlers"):
            stable_hash(figure)
    finally:
        pyplot.close(figure)


def test_seaborn_grid_is_unsupported() -> None:
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg", force=True)
    pyplot = pytest.importorskip("matplotlib.pyplot")
    pd = pytest.importorskip("pandas")
    seaborn = pytest.importorskip("seaborn")
    grid = seaborn.relplot(data=pd.DataFrame({"x": [0, 1], "y": [1, 3]}), x="x", y="y", kind="line")
    try:
        with pytest.raises(TypeError, match="explicit handlers"):
            stable_hash(grid)
    finally:
        pyplot.close(grid.figure)


def test_statsmodels_results_are_unsupported() -> None:
    np = pytest.importorskip("numpy")
    statsmodels = pytest.importorskip("statsmodels.api")

    x = np.arange(6, dtype=float)
    design = statsmodels.add_constant(x)
    y = np.array([1, 2, 3, 4, 5, 6], dtype=float)

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(statsmodels.OLS(y, design).fit())


def test_skimage_transform_is_unsupported() -> None:
    transform = pytest.importorskip("skimage.transform")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(transform.AffineTransform(scale=(1.0, 1.0), rotation=0.1, translation=(2, 3)))


def test_opencv_keypoint_is_unsupported() -> None:
    cv2 = pytest.importorskip("cv2")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(cv2.KeyPoint(10.0, 20.0, 5.0, 45.0, 0.7, 1, 2))


def test_spacy_handler() -> None:
    spacy = pytest.importorskip("spacy")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(spacy.blank("en"))


def test_torch_tensor_is_unsupported() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(torch.tensor([[1, 2], [3, 4]]))


def test_torch_module_is_unsupported() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(torch.nn.Linear(2, 3))


def test_pandas_dataframe_is_unsupported() -> None:
    pd = pytest.importorskip("pandas")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))


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


def test_polars_dataframe_is_unsupported() -> None:
    pl = pytest.importorskip("polars")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}))


def test_pyarrow_schema_handler() -> None:
    pa = pytest.importorskip("pyarrow")

    left = pa.schema([("a", pa.int64()), ("b", pa.string())])
    right = pa.schema([("a", pa.int64()), ("b", pa.string())])
    different = pa.schema([("a", pa.int32()), ("b", pa.string())])

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_pyarrow_table_is_unsupported() -> None:
    pa = pytest.importorskip("pyarrow")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(pa.table({"a": [1, 2], "b": ["x", "y"]}))


def test_xarray_dataset_is_unsupported() -> None:
    xr = pytest.importorskip("xarray")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(xr.Dataset({"values": ("row", [1, 2, 3])}))


def test_scipy_sparse_matrix_is_unsupported() -> None:
    sparse = pytest.importorskip("scipy.sparse")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(sparse.csr_matrix([[0, 1], [2, 0]]))


def test_pillow_image_is_unsupported() -> None:
    image_mod = pytest.importorskip("PIL.Image")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(image_mod.new("RGB", (2, 2), color=(10, 20, 30)))


def test_networkx_graph_is_unsupported() -> None:
    nx = pytest.importorskip("networkx")

    graph = nx.MultiDiGraph()
    graph.add_edge("a", "b")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(graph)


def test_rustworkx_graph_is_unsupported() -> None:
    rx = pytest.importorskip("rustworkx")

    graph = rx.PyDiGraph(multigraph=True)
    graph.add_node("a")
    graph.add_node("b")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(graph)


def test_sklearn_estimator_is_unsupported() -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(linear_model.LogisticRegression(C=1.0, max_iter=50, fit_intercept=True))


def test_xgboost_handler() -> None:
    xgb = pytest.importorskip("xgboost")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(xgb.XGBClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, random_state=7))


def test_lightgbm_handler() -> None:
    lgb = pytest.importorskip("lightgbm")

    with pytest.raises(TypeError, match="explicit handlers"):
        stable_hash(lgb.LGBMClassifier(num_leaves=31, n_estimators=10, learning_rate=0.1, random_state=7))
