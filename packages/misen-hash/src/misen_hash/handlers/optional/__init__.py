"""Aggregate optional handlers from per-library modules."""

from misen_hash.handler_base import HandlerTypeList, HandlerTypeRegistry
from misen_hash.handlers.optional.altair import altair_handlers, altair_handlers_by_type
from misen_hash.handlers.optional.attrs import attrs_handlers, attrs_handlers_by_type
from misen_hash.handlers.optional.dask import dask_handlers, dask_handlers_by_type
from misen_hash.handlers.optional.datasets import datasets_handlers, datasets_handlers_by_type
from misen_hash.handlers.optional.jax import jax_handlers, jax_handlers_by_type
from misen_hash.handlers.optional.lightgbm import lightgbm_handlers, lightgbm_handlers_by_type
from misen_hash.handlers.optional.matplotlib import matplotlib_handlers, matplotlib_handlers_by_type
from misen_hash.handlers.optional.msgspec import msgspec_handlers, msgspec_handlers_by_type
from misen_hash.handlers.optional.networkx import networkx_handlers, networkx_handlers_by_type
from misen_hash.handlers.optional.nltk import nltk_handlers, nltk_handlers_by_type
from misen_hash.handlers.optional.numpy import numpy_handlers, numpy_handlers_by_type
from misen_hash.handlers.optional.opencv import opencv_handlers, opencv_handlers_by_type
from misen_hash.handlers.optional.pandas import pandas_handlers, pandas_handlers_by_type
from misen_hash.handlers.optional.pillow import pillow_handlers, pillow_handlers_by_type
from misen_hash.handlers.optional.plotly import plotly_handlers, plotly_handlers_by_type
from misen_hash.handlers.optional.polars import polars_handlers, polars_handlers_by_type
from misen_hash.handlers.optional.pyarrow import pyarrow_handlers, pyarrow_handlers_by_type
from misen_hash.handlers.optional.pydantic import pydantic_handlers, pydantic_handlers_by_type
from misen_hash.handlers.optional.rustworkx import rustworkx_handlers, rustworkx_handlers_by_type
from misen_hash.handlers.optional.scipy import scipy_handlers, scipy_handlers_by_type
from misen_hash.handlers.optional.seaborn import seaborn_handlers, seaborn_handlers_by_type
from misen_hash.handlers.optional.sentencepiece import sentencepiece_handlers, sentencepiece_handlers_by_type
from misen_hash.handlers.optional.skimage import skimage_handlers, skimage_handlers_by_type
from misen_hash.handlers.optional.sklearn import sklearn_handlers, sklearn_handlers_by_type
from misen_hash.handlers.optional.spacy import spacy_handlers, spacy_handlers_by_type
from misen_hash.handlers.optional.statsmodels import statsmodels_handlers, statsmodels_handlers_by_type
from misen_hash.handlers.optional.sympy import sympy_handlers, sympy_handlers_by_type
from misen_hash.handlers.optional.tensorflow import tensorflow_handlers, tensorflow_handlers_by_type
from misen_hash.handlers.optional.tokenizers import tokenizers_handlers, tokenizers_handlers_by_type
from misen_hash.handlers.optional.torch import torch_handlers, torch_handlers_by_type
from misen_hash.handlers.optional.transformers import transformers_handlers, transformers_handlers_by_type
from misen_hash.handlers.optional.xarray import xarray_handlers, xarray_handlers_by_type
from misen_hash.handlers.optional.xgboost import xgboost_handlers, xgboost_handlers_by_type

__all__ = ["optional_handlers", "optional_handlers_by_type"]

optional_handlers: HandlerTypeList = [
    *msgspec_handlers,
    *altair_handlers,
    *attrs_handlers,
    *dask_handlers,
    *datasets_handlers,
    *jax_handlers,
    *matplotlib_handlers,
    *seaborn_handlers,
    *numpy_handlers,
    *opencv_handlers,
    *nltk_handlers,
    *pandas_handlers,
    *pydantic_handlers,
    *tokenizers_handlers,
    *transformers_handlers,
    *polars_handlers,
    *pyarrow_handlers,
    *xarray_handlers,
    *scipy_handlers,
    *sentencepiece_handlers,
    *sympy_handlers,
    *tensorflow_handlers,
    *pillow_handlers,
    *plotly_handlers,
    *spacy_handlers,
    *networkx_handlers,
    *statsmodels_handlers,
    *skimage_handlers,
    *rustworkx_handlers,
    *xgboost_handlers,
    *lightgbm_handlers,
    *sklearn_handlers,
    *torch_handlers,
]

optional_handlers_by_type: HandlerTypeRegistry = {
    **msgspec_handlers_by_type,
    **altair_handlers_by_type,
    **attrs_handlers_by_type,
    **dask_handlers_by_type,
    **datasets_handlers_by_type,
    **jax_handlers_by_type,
    **matplotlib_handlers_by_type,
    **seaborn_handlers_by_type,
    **numpy_handlers_by_type,
    **opencv_handlers_by_type,
    **nltk_handlers_by_type,
    **pandas_handlers_by_type,
    **pydantic_handlers_by_type,
    **tokenizers_handlers_by_type,
    **transformers_handlers_by_type,
    **polars_handlers_by_type,
    **pyarrow_handlers_by_type,
    **xarray_handlers_by_type,
    **scipy_handlers_by_type,
    **sentencepiece_handlers_by_type,
    **sympy_handlers_by_type,
    **tensorflow_handlers_by_type,
    **pillow_handlers_by_type,
    **plotly_handlers_by_type,
    **spacy_handlers_by_type,
    **networkx_handlers_by_type,
    **statsmodels_handlers_by_type,
    **skimage_handlers_by_type,
    **rustworkx_handlers_by_type,
    **xgboost_handlers_by_type,
    **lightgbm_handlers_by_type,
    **sklearn_handlers_by_type,
    **torch_handlers_by_type,
}
