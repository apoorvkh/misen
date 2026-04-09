"""Aggregate optional serializers for third-party library types."""

from misen.utils.serde.serializer_base import SerializerTypeList, SerializerTypeRegistry
from misen.utils.serde.serializers.optional.msgspec_struct import (
    msgspec_struct_serializers,
    msgspec_struct_serializers_by_type,
)
from misen.utils.serde.serializers.optional.numpy import numpy_serializers, numpy_serializers_by_type
from misen.utils.serde.serializers.optional.pandas import pandas_serializers, pandas_serializers_by_type
from misen.utils.serde.serializers.optional.pillow import pillow_serializers, pillow_serializers_by_type
from misen.utils.serde.serializers.optional.polars import polars_serializers, polars_serializers_by_type
from misen.utils.serde.serializers.optional.pyarrow import pyarrow_serializers, pyarrow_serializers_by_type
from misen.utils.serde.serializers.optional.pydantic import pydantic_serializers, pydantic_serializers_by_type
from misen.utils.serde.serializers.optional.scipy import scipy_serializers, scipy_serializers_by_type
from misen.utils.serde.serializers.optional.torch import torch_serializers, torch_serializers_by_type

__all__ = ["optional_serializers", "optional_serializers_by_type"]

optional_serializers: SerializerTypeList = [
    *msgspec_struct_serializers,
    *pydantic_serializers,
    *numpy_serializers,
    *torch_serializers,
    *pandas_serializers,
    *polars_serializers,
    *pyarrow_serializers,
    *scipy_serializers,
    *pillow_serializers,
]

optional_serializers_by_type: SerializerTypeRegistry = {
    **msgspec_struct_serializers_by_type,
    **pydantic_serializers_by_type,
    **numpy_serializers_by_type,
    **torch_serializers_by_type,
    **pandas_serializers_by_type,
    **polars_serializers_by_type,
    **pyarrow_serializers_by_type,
    **scipy_serializers_by_type,
    **pillow_serializers_by_type,
}
