"""Serializers for HuggingFace datasets."""

import importlib.util
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import (
    Serializer,
    SerializerTypeRegistry,
)

__all__ = ["hf_datasets_serializers", "hf_datasets_serializers_by_type"]

hf_datasets_serializers: list[type[Serializer]] = []
hf_datasets_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("datasets") is not None:
    from pathlib import Path

    class HFDatasetSerializer(Serializer[Any]):
        """Serialize HuggingFace ``Dataset`` via Arrow-based ``save_to_disk``."""

        @staticmethod
        def match(obj: Any) -> bool:
            from datasets import Dataset

            return isinstance(obj, Dataset)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import datasets

            obj.save_to_disk(str(directory / "dataset"))
            return {"datasets_version": datasets.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            from datasets import Dataset

            return Dataset.load_from_disk(str(directory / "dataset"))

    class HFDatasetDictSerializer(Serializer[Any]):
        """Serialize HuggingFace ``DatasetDict`` via Arrow-based ``save_to_disk``."""

        @staticmethod
        def match(obj: Any) -> bool:
            from datasets import DatasetDict

            return isinstance(obj, DatasetDict)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import datasets

            obj.save_to_disk(str(directory / "dataset_dict"))
            return {"datasets_version": datasets.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            from datasets import DatasetDict

            return DatasetDict.load_from_disk(str(directory / "dataset_dict"))

    hf_datasets_serializers = [HFDatasetDictSerializer, HFDatasetSerializer]
    hf_datasets_serializers_by_type = {
        "datasets.dataset_dict.DatasetDict": HFDatasetDictSerializer,
        "datasets.arrow_dataset.Dataset": HFDatasetSerializer,
    }
