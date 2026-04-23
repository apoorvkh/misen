"""Aggregated v2 serializer registry.

Dispatch order (linear ``match`` scan after the ``TypeDispatchRegistry``
fast paths):

1. ``MsgpackLeafSerializer`` (predicate-gated) — claims any value
   whose whole subtree is msgpack-native so primitive structures
   collapse to a single leaf.
2. Recursive container serializers (``DictSerializer``, ...) — handle
   structures that contain at least one non-msgpack-native value by
   recursing so each child dispatches independently.
3. Library-specific leaf / directory serializers (numpy, torch,
   pandas, sklearn, ...) — each with a strict ``isinstance`` match.

Every library module exports ``<name>_serializers`` and
``<name>_serializers_by_type``; the latter goes into the by-type fast
path of :class:`TypeDispatchRegistry`.  Only stdlib contributes to
``all_volatile_types`` — library-specific types always dispatch the
same way regardless of contents.
"""

from misen.utils.serde.base import BaseSerializer
from misen.utils.serde.libs.altair import altair_serializers, altair_serializers_by_type
from misen.utils.serde.libs.attrs import attrs_serializers, attrs_serializers_by_type
from misen.utils.serde.libs.catboost import catboost_serializers, catboost_serializers_by_type
from misen.utils.serde.libs.dataclass import dataclass_serializers, dataclass_serializers_by_type
from misen.utils.serde.libs.faiss import faiss_serializers, faiss_serializers_by_type
from misen.utils.serde.libs.geopandas import geopandas_serializers, geopandas_serializers_by_type
from misen.utils.serde.libs.hf_datasets import hf_datasets_serializers, hf_datasets_serializers_by_type
from misen.utils.serde.libs.jax import jax_serializers, jax_serializers_by_type
from misen.utils.serde.libs.keras import keras_serializers, keras_serializers_by_type
from misen.utils.serde.libs.lightgbm import lightgbm_serializers, lightgbm_serializers_by_type
from misen.utils.serde.libs.msgspec_struct import (
    msgspec_struct_serializers,
    msgspec_struct_serializers_by_type,
)
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
from misen.utils.serde.libs.stdlib import (
    stdlib_serializers,
    stdlib_serializers_by_type,
    stdlib_volatile_types,
)
from misen.utils.serde.libs.sympy import sympy_serializers, sympy_serializers_by_type
from misen.utils.serde.libs.tensorflow import tensorflow_serializers, tensorflow_serializers_by_type
from misen.utils.serde.libs.tokenizers import tokenizers_serializers, tokenizers_serializers_by_type
from misen.utils.serde.libs.torch import torch_serializers, torch_serializers_by_type
from misen.utils.serde.libs.transformers import (
    transformers_serializers,
    transformers_serializers_by_type,
)
from misen.utils.serde.libs.xarray import xarray_serializers, xarray_serializers_by_type
from misen.utils.serde.libs.xgboost import xgboost_serializers, xgboost_serializers_by_type
from misen.utils.serde.registry import Registry

__all__ = [
    "all_serializers",
    "all_serializers_by_type",
    "all_volatile_types",
    "default_registry",
]


# ``stdlib_serializers`` is already ordered Msgpack-first, then
# containers.  Library-specific serializers sit after the containers —
# their matches are all strict ``isinstance`` / ``hasattr`` checks so
# they never conflict with the general stdlib path.  The grouping
# mirrors v1's ordering for readability; within a group the order
# doesn't matter (matches are disjoint).
all_serializers: list[type[BaseSerializer]] = [
    *stdlib_serializers,
    # Structured data — class-level protocol checks.
    *msgspec_struct_serializers,
    *pydantic_serializers,
    *attrs_serializers,
    *dataclass_serializers,
    # Scientific / multidimensional.
    *xarray_serializers,
    *sympy_serializers,
    # Numeric arrays.
    *jax_serializers,
    *numpy_serializers,
    *tensorflow_serializers,
    *torch_serializers,
    # Tabular data.
    *pandas_serializers,
    *polars_serializers,
    *pyarrow_serializers,
    *geopandas_serializers,
    # Sparse.
    *scipy_serializers,
    # Geospatial.
    *shapely_serializers,
    # ML models.
    *keras_serializers,
    *sklearn_serializers,
    *xgboost_serializers,
    *lightgbm_serializers,
    *catboost_serializers,
    *onnx_serializers,
    *faiss_serializers,
    # NLP / tokenization.
    *tokenizers_serializers,
    # HuggingFace.
    *transformers_serializers,
    *hf_datasets_serializers,
    # Visualization / media.
    *altair_serializers,
    *plotly_serializers,
    *pillow_serializers,
]


all_serializers_by_type: dict[str, type[BaseSerializer]] = {
    **stdlib_serializers_by_type,
    **msgspec_struct_serializers_by_type,
    **pydantic_serializers_by_type,
    **attrs_serializers_by_type,
    **dataclass_serializers_by_type,
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


# Only stdlib contributes volatile types — library-specific classes
# (Tensor, DataFrame, BaseModel, ...) always dispatch the same way.
all_volatile_types: frozenset[type] = frozenset(stdlib_volatile_types)


_default_registry: Registry | None = None


def default_registry() -> Registry:
    """Return a shared :class:`Registry` built from :data:`all_serializers`."""
    global _default_registry  # noqa: PLW0603
    if _default_registry is None:
        _default_registry = Registry(
            all_serializers,
            by_type_name=all_serializers_by_type,
            volatile_types=all_volatile_types,
        )
    return _default_registry
