import dataclasses
import functools
import importlib.util
import ipaddress
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


def test_altair_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("altair") is None:
        assert all(handler.__name__ != "AltairChartHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.altair import AltairChartHandler

        assert AltairChartHandler in optional_handlers


def test_matplotlib_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("matplotlib") is None:
        assert all(handler.__name__ != "MatplotlibFigureHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.matplotlib import MatplotlibFigureHandler

        assert MatplotlibFigureHandler in optional_handlers


def test_seaborn_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("seaborn") is None:
        assert all(handler.__name__ != "SeabornGridHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.seaborn import SeabornGridHandler

        assert SeabornGridHandler in optional_handlers


def test_statsmodels_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("statsmodels") is None:
        assert all(handler.__name__ != "StatsmodelsResultsHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.statsmodels import StatsmodelsResultsHandler

        assert StatsmodelsResultsHandler in optional_handlers


def test_skimage_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("skimage") is None:
        assert all(handler.__name__ != "SkimageTransformHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.skimage import SkimageTransformHandler

        assert SkimageTransformHandler in optional_handlers


def test_opencv_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("cv2") is None:
        assert all(handler.__name__ != "OpenCVKeyPointHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.opencv import OpenCVKeyPointHandler

        assert OpenCVKeyPointHandler in optional_handlers


def test_nltk_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("nltk") is None:
        assert all(handler.__name__ != "NLTKTreeHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.nltk import NLTKTreeHandler

        assert NLTKTreeHandler in optional_handlers


def test_dask_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("dask") is None:
        assert all(handler.__name__ != "DaskCollectionHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.dask import DaskCollectionHandler

        assert DaskCollectionHandler in optional_handlers


def test_datasets_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("datasets") is None:
        assert all(handler.__name__ != "HFDatasetHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.datasets import HFDatasetHandler

        assert HFDatasetHandler in optional_handlers


def test_transformers_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("transformers") is None:
        assert all(handler.__name__ != "TransformersConfigHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.transformers import TransformersConfigHandler

        assert TransformersConfigHandler in optional_handlers


def test_tokenizers_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("tokenizers") is None:
        assert all(handler.__name__ != "HFTokenizerHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.tokenizers import HFTokenizerHandler

        assert HFTokenizerHandler in optional_handlers


def test_sentencepiece_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("sentencepiece") is None:
        assert all(handler.__name__ != "SentencePieceProcessorHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.sentencepiece import SentencePieceProcessorHandler

        assert SentencePieceProcessorHandler in optional_handlers


def test_sympy_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("sympy") is None:
        assert all(handler.__name__ != "SymPyBasicHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.sympy import SymPyBasicHandler

        assert SymPyBasicHandler in optional_handlers


def test_plotly_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("plotly") is None:
        assert all(handler.__name__ != "PlotlyFigureHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.plotly import PlotlyFigureHandler

        assert PlotlyFigureHandler in optional_handlers


def test_spacy_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("spacy") is None:
        assert all(handler.__name__ != "SpacyLanguageHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.spacy import SpacyLanguageHandler

        assert SpacyLanguageHandler in optional_handlers


def test_jax_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("jax") is None:
        assert all(handler.__name__ != "JaxArrayHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.jax import JaxArrayHandler

        assert JaxArrayHandler in optional_handlers


def test_tensorflow_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("tensorflow") is None:
        assert all(handler.__name__ != "TensorFlowTensorHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.tensorflow import TensorFlowTensorHandler

        assert TensorFlowTensorHandler in optional_handlers


def test_xgboost_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("xgboost") is None:
        assert all(handler.__name__ != "XGBoostBoosterHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.xgboost import XGBoostBoosterHandler

        assert XGBoostBoosterHandler in optional_handlers


def test_lightgbm_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("lightgbm") is None:
        assert all(handler.__name__ != "LightGBMBoosterHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.lightgbm import LightGBMBoosterHandler

        assert LightGBMBoosterHandler in optional_handlers


def test_rustworkx_handlers_are_registered_as_optional() -> None:
    if importlib.util.find_spec("rustworkx") is None:
        assert all(handler.__name__ != "RustworkxGraphHandler" for handler in optional_handlers)
    else:
        from misen_hash.handlers.optional.rustworkx import RustworkxGraphHandler

        assert RustworkxGraphHandler in optional_handlers


def test_stable_hash_falls_back_to_dill_handler() -> None:
    assert isinstance(stable_hash(_Custom()), int)


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
        (functools.partial(pow, 2, exp=5), functools.partial(pow, 2, exp=5)),
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


def test_partial_hash_includes_instance_attributes() -> None:
    left = functools.partial(pow, 2, exp=5)
    right = functools.partial(pow, 2, exp=5)
    different = functools.partial(pow, 2, exp=5)

    left.meta = {"tags": ["a", "b"]}
    right.meta = {"tags": ["a", "b"]}
    different.meta = {"tags": ["a", "c"]}

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


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


def test_numpy_handlers_cover_arrays_and_scalars() -> None:
    np = pytest.importorskip("numpy")

    numeric_array = np.array([[1, 2], [3, 4]], dtype=np.int64)
    assert stable_hash(numeric_array) == stable_hash(np.array([[1, 2], [3, 4]], dtype=np.int64))
    assert stable_hash(numeric_array) != stable_hash(np.array([[1, 2], [3, 4]], dtype=np.int32))

    object_array = np.array([{"a": [1, 2]}, [3, 4]], dtype=object)
    assert stable_hash(object_array) == stable_hash(np.array([{"a": [1, 2]}, [3, 4]], dtype=object))
    assert stable_hash(np.int64(7)) != stable_hash(np.int32(7))


def test_jax_array_handler() -> None:
    jnp = pytest.importorskip("jax.numpy")

    left = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    right = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    different = jnp.array([[1, 2], [3, 4]], dtype=jnp.int64)

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_tensorflow_tensor_handler() -> None:
    tf = pytest.importorskip("tensorflow")

    left = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
    right = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
    different = tf.constant([[1, 2], [3, 4]], dtype=tf.int64)

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_dask_handler() -> None:
    dask_array = pytest.importorskip("dask.array")

    left = dask_array.arange(10, chunks=(5,)).reshape((2, 5))
    right = dask_array.arange(10, chunks=(5,)).reshape((2, 5))
    different = dask_array.arange(10, chunks=(10,)).reshape((2, 5))

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_datasets_handler() -> None:
    datasets = pytest.importorskip("datasets")

    left = datasets.Dataset.from_dict({"x": [1, 2], "y": ["a", "b"]})
    right = datasets.Dataset.from_dict({"x": [1, 2], "y": ["a", "b"]})
    different = datasets.Dataset.from_dict({"x": [1, 3], "y": ["a", "b"]})

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_transformers_config_handler() -> None:
    transformers = pytest.importorskip("transformers")

    left = transformers.BertConfig(hidden_size=16, num_attention_heads=2, intermediate_size=32)
    right = transformers.BertConfig(hidden_size=16, num_attention_heads=2, intermediate_size=32)
    different = transformers.BertConfig(hidden_size=32, num_attention_heads=2, intermediate_size=32)

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_tokenizers_handler() -> None:
    tokenizers = pytest.importorskip("tokenizers")

    left = tokenizers.Tokenizer.from_str(
        """
        {"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,
         "pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,
         "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"world":2},"unk_token":"[UNK]"}}
        """
    )
    right = tokenizers.Tokenizer.from_str(
        """
        {"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,
         "pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,
         "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"world":2},"unk_token":"[UNK]"}}
        """
    )
    different = tokenizers.Tokenizer.from_str(
        """
        {"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,
         "pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,
         "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"mars":2},"unk_token":"[UNK]"}}
        """
    )

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_sentencepiece_handler() -> None:
    pytest.importorskip("sentencepiece")
    pytest.skip("SentencePiece model construction requires external model artifacts.")


def test_sympy_handler() -> None:
    sympy = pytest.importorskip("sympy")

    x = sympy.Symbol("x")
    left = (x + 1) ** 2
    right = (x + 1) ** 2
    different = (x + 2) ** 2

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_plotly_handler() -> None:
    graph_objects = pytest.importorskip("plotly.graph_objects")

    left = graph_objects.Figure(data=[graph_objects.Scatter(y=[1, 2, 3])])
    right = graph_objects.Figure(data=[graph_objects.Scatter(y=[1, 2, 3])])
    different = graph_objects.Figure(data=[graph_objects.Scatter(y=[1, 2, 4])])

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_altair_handler() -> None:
    altair = pytest.importorskip("altair")
    pd = pytest.importorskip("pandas")

    left_data = pd.DataFrame({"x": [0, 1, 2], "y": [1, 3, 5]})
    right_data = pd.DataFrame({"x": [0, 1, 2], "y": [1, 3, 5]})
    different_data = pd.DataFrame({"x": [0, 1, 2], "y": [1, 3, 6]})

    left = altair.Chart(left_data).mark_line().encode(x="x:Q", y="y:Q")
    right = altair.Chart(right_data).mark_line().encode(x="x:Q", y="y:Q")
    different = altair.Chart(different_data).mark_line().encode(x="x:Q", y="y:Q")

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_matplotlib_handler() -> None:
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg", force=True)
    pyplot = pytest.importorskip("matplotlib.pyplot")

    left, left_ax = pyplot.subplots()
    left_ax.plot([0, 1, 2], [1, 3, 5], label="line")
    left_ax.scatter([0, 1, 2], [1, 3, 5], c=[1, 2, 3])

    right, right_ax = pyplot.subplots()
    right_ax.plot([0, 1, 2], [1, 3, 5], label="line")
    right_ax.scatter([0, 1, 2], [1, 3, 5], c=[1, 2, 3])

    different, different_ax = pyplot.subplots()
    different_ax.plot([0, 1, 2], [1, 3, 6], label="line")
    different_ax.scatter([0, 1, 2], [1, 3, 6], c=[1, 2, 3])

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)
    assert stable_hash(left_ax) == stable_hash(right_ax)

    pyplot.close(left)
    pyplot.close(right)
    pyplot.close(different)


def test_seaborn_handler() -> None:
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg", force=True)
    pyplot = pytest.importorskip("matplotlib.pyplot")
    pd = pytest.importorskip("pandas")
    seaborn = pytest.importorskip("seaborn")

    left_data = pd.DataFrame({"x": [0, 1, 2], "y": [1, 3, 5], "group": ["a", "a", "a"]})
    right_data = pd.DataFrame({"x": [0, 1, 2], "y": [1, 3, 5], "group": ["a", "a", "a"]})
    different_data = pd.DataFrame({"x": [0, 1, 2], "y": [1, 3, 6], "group": ["a", "a", "a"]})

    left = seaborn.relplot(data=left_data, x="x", y="y", kind="line", col="group")
    right = seaborn.relplot(data=right_data, x="x", y="y", kind="line", col="group")
    different = seaborn.relplot(data=different_data, x="x", y="y", kind="line", col="group")

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)

    pyplot.close(left.figure)
    pyplot.close(right.figure)
    pyplot.close(different.figure)


def test_statsmodels_handler() -> None:
    np = pytest.importorskip("numpy")
    statsmodels = pytest.importorskip("statsmodels.api")

    x = np.arange(6, dtype=float)
    design = statsmodels.add_constant(x)
    y = np.array([1, 2, 3, 4, 5, 6], dtype=float)

    left = statsmodels.OLS(y, design).fit()
    right = statsmodels.OLS(y, design).fit()
    different = statsmodels.OLS(y + np.array([0, 0, 0, 0, 0, 1], dtype=float), design).fit()

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_skimage_handler() -> None:
    transform = pytest.importorskip("skimage.transform")

    left = transform.AffineTransform(scale=(1.0, 1.0), rotation=0.1, translation=(2, 3))
    right = transform.AffineTransform(scale=(1.0, 1.0), rotation=0.1, translation=(2, 3))
    different = transform.AffineTransform(scale=(1.1, 1.0), rotation=0.1, translation=(2, 3))

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_opencv_handler() -> None:
    cv2 = pytest.importorskip("cv2")

    left = cv2.KeyPoint(10.0, 20.0, 5.0, 45.0, 0.7, 1, 2)
    right = cv2.KeyPoint(10.0, 20.0, 5.0, 45.0, 0.7, 1, 2)
    different = cv2.KeyPoint(10.0, 20.0, 5.0, 46.0, 0.7, 1, 2)

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_nltk_handler() -> None:
    nltk = pytest.importorskip("nltk")

    left_tree = nltk.Tree("S", [nltk.Tree("NP", ["I"]), nltk.Tree("VP", ["run"])])
    right_tree = nltk.Tree("S", [nltk.Tree("NP", ["I"]), nltk.Tree("VP", ["run"])])
    different_tree = nltk.Tree("S", [nltk.Tree("NP", ["I"]), nltk.Tree("VP", ["walk"])])

    left_cfg = nltk.CFG.fromstring("S -> NP VP\nNP -> 'I'\nVP -> 'run'")
    right_cfg = nltk.CFG.fromstring("S -> NP VP\nNP -> 'I'\nVP -> 'run'")
    different_cfg = nltk.CFG.fromstring("S -> NP VP\nNP -> 'I'\nVP -> 'walk'")

    assert stable_hash(left_tree) == stable_hash(right_tree)
    assert stable_hash(left_tree) != stable_hash(different_tree)
    assert stable_hash(left_cfg) == stable_hash(right_cfg)
    assert stable_hash(left_cfg) != stable_hash(different_cfg)


def test_spacy_handler() -> None:
    spacy = pytest.importorskip("spacy")

    left = spacy.blank("en")
    right = spacy.blank("en")
    different = spacy.blank("de")

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


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


def test_rustworkx_handler() -> None:
    rx = pytest.importorskip("rustworkx")

    first = rx.PyDiGraph(multigraph=True)
    first_a = first.add_node(("a", [1, 2]))
    first_b = first.add_node(("b", [3]))
    first.add_edge(first_a, first_b, {"weight": 5})

    second = rx.PyDiGraph(multigraph=True)
    second_b = second.add_node(("b", [3]))
    second_a = second.add_node(("a", [1, 2]))
    second.add_edge(second_a, second_b, {"weight": 5})

    different = rx.PyDiGraph(multigraph=True)
    different_a = different.add_node(("a", [1, 2]))
    different_b = different.add_node(("b", [3]))
    different.add_edge(different_a, different_b, {"weight": 6})

    assert stable_hash(first) == stable_hash(second)
    assert stable_hash(first) != stable_hash(different)


def test_sklearn_handler() -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")

    left = linear_model.LogisticRegression(C=1.0, max_iter=50, fit_intercept=True)
    right = linear_model.LogisticRegression(C=1.0, max_iter=50, fit_intercept=True)
    different = linear_model.LogisticRegression(C=0.9, max_iter=50, fit_intercept=True)

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_xgboost_handler() -> None:
    xgb = pytest.importorskip("xgboost")

    left = xgb.XGBClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, random_state=7)
    right = xgb.XGBClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, random_state=7)
    different = xgb.XGBClassifier(max_depth=4, n_estimators=10, learning_rate=0.1, random_state=7)

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)


def test_lightgbm_handler() -> None:
    lgb = pytest.importorskip("lightgbm")

    left = lgb.LGBMClassifier(num_leaves=31, n_estimators=10, learning_rate=0.1, random_state=7)
    right = lgb.LGBMClassifier(num_leaves=31, n_estimators=10, learning_rate=0.1, random_state=7)
    different = lgb.LGBMClassifier(num_leaves=15, n_estimators=10, learning_rate=0.1, random_state=7)

    assert stable_hash(left) == stable_hash(right)
    assert stable_hash(left) != stable_hash(different)
