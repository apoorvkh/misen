"""Aggregated registry of all :class:`Serializer` subclasses in ``libs/``.

Each submodule under ``libs/`` defines a ``<lib>_serializers`` list and a
``<lib>_serializers_by_type`` dict. This module concatenates them into
``all_serializers`` (ordered by dispatch priority) and
``all_serializers_by_type`` (keyed by exact qualified type name) for
:mod:`misen.utils.serde.registry` to consume.

Adding a new serializer library: create ``libs/<name>.py`` exposing
``<name>_serializers`` and ``<name>_serializers_by_type``, then import
them here. Order in ``all_serializers`` determines priority for
:meth:`Serializer.match`-based dispatch — catch-all serializers must
come last.
"""

from misen.utils.serde.base import Serializer, SerializerTypeRegistry
from misen.utils.serde.libs.altair import altair_serializers, altair_serializers_by_type
from misen.utils.serde.libs.attrs import attrs_serializers, attrs_serializers_by_type
from misen.utils.serde.libs.catboost import catboost_serializers, catboost_serializers_by_type
from misen.utils.serde.libs.faiss import faiss_serializers, faiss_serializers_by_type
from misen.utils.serde.libs.geopandas import geopandas_serializers, geopandas_serializers_by_type
from misen.utils.serde.libs.hf_datasets import hf_datasets_serializers, hf_datasets_serializers_by_type
from misen.utils.serde.libs.jax import jax_serializers, jax_serializers_by_type
from misen.utils.serde.libs.keras import keras_serializers, keras_serializers_by_type
from misen.utils.serde.libs.lightgbm import lightgbm_serializers, lightgbm_serializers_by_type
from misen.utils.serde.libs.msgspec_struct import msgspec_struct_serializers, msgspec_struct_serializers_by_type
from misen.utils.serde.libs.numpy import numpy_serializers, numpy_serializers_by_type
from misen.utils.serde.libs.onnx import onnx_serializers, onnx_serializers_by_type
from misen.utils.serde.libs.pandas import pandas_serializers, pandas_serializers_by_type
from misen.utils.serde.libs.pillow import pillow_serializers, pillow_serializers_by_type
from misen.utils.serde.libs.plotly import plotly_serializers, plotly_serializers_by_type
from misen.utils.serde.libs.polars import polars_serializers, polars_serializers_by_type
from misen.utils.serde.libs.pyarrow import pyarrow_serializers, pyarrow_serializers_by_type
from misen.utils.serde.libs.pydantic import pydantic_serializers, pydantic_serializers_by_type
from misen.utils.serde.libs.scipy import scipy_serializers, scipy_serializers_by_type
from misen.utils.serde.libs.shapely import shapely_serializers, shapely_serializers_by_type
from misen.utils.serde.libs.sklearn import sklearn_serializers, sklearn_serializers_by_type
from misen.utils.serde.libs.stdlib import stdlib_serializers, stdlib_serializers_by_type
from misen.utils.serde.libs.sympy import sympy_serializers, sympy_serializers_by_type
from misen.utils.serde.libs.tensorflow import tensorflow_serializers, tensorflow_serializers_by_type
from misen.utils.serde.libs.tokenizers import tokenizers_serializers, tokenizers_serializers_by_type
from misen.utils.serde.libs.torch import torch_serializers, torch_serializers_by_type
from misen.utils.serde.libs.transformers import transformers_serializers, transformers_serializers_by_type
from misen.utils.serde.libs.xarray import xarray_serializers, xarray_serializers_by_type
from misen.utils.serde.libs.xgboost import xgboost_serializers, xgboost_serializers_by_type

__all__ = ["all_serializers", "all_serializers_by_type"]

all_serializers: list[type[Serializer]] = [
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
    ### Standard library
    *stdlib_serializers,  # includes catch-all, must be last
]

all_serializers_by_type: SerializerTypeRegistry = {
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
    **stdlib_serializers_by_type,
}
