"""Aggregate all serializer registries."""

from misen.utils.serde.serializer_base import SerializerTypeList, SerializerTypeRegistry
from misen.utils.serde.serializers.optional import optional_serializers, optional_serializers_by_type
from misen.utils.serde.serializers.stdlib import stdlib_serializers, stdlib_serializers_by_type

__all__ = [
    "optional_serializers",
    "optional_serializers_by_type",
    "stdlib_serializers",
    "stdlib_serializers_by_type",
]

all_serializers: SerializerTypeList = [*optional_serializers, *stdlib_serializers]
all_serializers_by_type: SerializerTypeRegistry = {**stdlib_serializers_by_type, **optional_serializers_by_type}
