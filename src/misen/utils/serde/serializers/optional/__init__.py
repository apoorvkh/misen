"""Aggregate optional serializers for third-party library types."""

from misen.utils.serde.serializer_base import SerializerTypeList, SerializerTypeRegistry
from misen.utils.serde.serializers.optional.altair import altair_serializers, altair_serializers_by_type
from misen.utils.serde.serializers.optional.attrs import attrs_serializers, attrs_serializers_by_type
from misen.utils.serde.serializers.optional.catboost import catboost_serializers, catboost_serializers_by_type
from misen.utils.serde.serializers.optional.faiss import faiss_serializers, faiss_serializers_by_type
from misen.utils.serde.serializers.optional.geopandas import geopandas_serializers, geopandas_serializers_by_type
from misen.utils.serde.serializers.optional.hf_datasets import hf_datasets_serializers, hf_datasets_serializers_by_type
from misen.utils.serde.serializers.optional.jax import jax_serializers, jax_serializers_by_type
from misen.utils.serde.serializers.optional.keras import keras_serializers, keras_serializers_by_type
from misen.utils.serde.serializers.optional.lightgbm import lightgbm_serializers, lightgbm_serializers_by_type
from misen.utils.serde.serializers.optional.msgspec_struct import (
    msgspec_struct_serializers,
    msgspec_struct_serializers_by_type,
)
from misen.utils.serde.serializers.optional.numpy import numpy_serializers, numpy_serializers_by_type
from misen.utils.serde.serializers.optional.onnx import onnx_serializers, onnx_serializers_by_type
from misen.utils.serde.serializers.optional.pandas import pandas_serializers, pandas_serializers_by_type
from misen.utils.serde.serializers.optional.pillow import pillow_serializers, pillow_serializers_by_type
from misen.utils.serde.serializers.optional.plotly import plotly_serializers, plotly_serializers_by_type
from misen.utils.serde.serializers.optional.polars import polars_serializers, polars_serializers_by_type
from misen.utils.serde.serializers.optional.pyarrow import pyarrow_serializers, pyarrow_serializers_by_type
from misen.utils.serde.serializers.optional.pydantic import pydantic_serializers, pydantic_serializers_by_type
from misen.utils.serde.serializers.optional.scipy import scipy_serializers, scipy_serializers_by_type
from misen.utils.serde.serializers.optional.shapely import shapely_serializers, shapely_serializers_by_type
from misen.utils.serde.serializers.optional.sklearn import sklearn_serializers, sklearn_serializers_by_type
from misen.utils.serde.serializers.optional.sympy import sympy_serializers, sympy_serializers_by_type
from misen.utils.serde.serializers.optional.tensorflow import tensorflow_serializers, tensorflow_serializers_by_type
from misen.utils.serde.serializers.optional.tokenizers import tokenizers_serializers, tokenizers_serializers_by_type
from misen.utils.serde.serializers.optional.torch import torch_serializers, torch_serializers_by_type
from misen.utils.serde.serializers.optional.transformers import (
    transformers_serializers,
    transformers_serializers_by_type,
)
from misen.utils.serde.serializers.optional.xarray import xarray_serializers, xarray_serializers_by_type
from misen.utils.serde.serializers.optional.xgboost import xgboost_serializers, xgboost_serializers_by_type

__all__ = ["optional_serializers", "optional_serializers_by_type"]

optional_serializers: SerializerTypeList = [
    # Structured data
    *msgspec_struct_serializers,
    *pydantic_serializers,
    *attrs_serializers,
    # Scientific / multidimensional
    *xarray_serializers,
    *sympy_serializers,
    # Numeric arrays
    *jax_serializers,
    *numpy_serializers,
    *tensorflow_serializers,
    *torch_serializers,
    # Tabular data
    *pandas_serializers,
    *polars_serializers,
    *pyarrow_serializers,
    *geopandas_serializers,
    # Sparse
    *scipy_serializers,
    # Geospatial
    *shapely_serializers,
    # ML models
    *keras_serializers,
    *sklearn_serializers,
    *xgboost_serializers,
    *lightgbm_serializers,
    *catboost_serializers,
    *onnx_serializers,
    *faiss_serializers,
    # NLP / tokenization
    *tokenizers_serializers,
    # HuggingFace
    *transformers_serializers,
    *hf_datasets_serializers,
    # Visualization
    *altair_serializers,
    *plotly_serializers,
    *pillow_serializers,
]

optional_serializers_by_type: SerializerTypeRegistry = {
    **msgspec_struct_serializers_by_type,
    **pydantic_serializers_by_type,
    **attrs_serializers_by_type,
    **xarray_serializers_by_type,
    **sympy_serializers_by_type,
    **jax_serializers_by_type,
    **numpy_serializers_by_type,
    **tensorflow_serializers_by_type,
    **torch_serializers_by_type,
    **pandas_serializers_by_type,
    **polars_serializers_by_type,
    **pyarrow_serializers_by_type,
    **geopandas_serializers_by_type,
    **scipy_serializers_by_type,
    **shapely_serializers_by_type,
    **keras_serializers_by_type,
    **sklearn_serializers_by_type,
    **xgboost_serializers_by_type,
    **lightgbm_serializers_by_type,
    **catboost_serializers_by_type,
    **onnx_serializers_by_type,
    **faiss_serializers_by_type,
    **tokenizers_serializers_by_type,
    **transformers_serializers_by_type,
    **hf_datasets_serializers_by_type,
    **altair_serializers_by_type,
    **plotly_serializers_by_type,
    **pillow_serializers_by_type,
}
