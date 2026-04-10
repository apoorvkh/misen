"""Serializers for HuggingFace datasets."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["hf_datasets_serializers", "hf_datasets_serializers_by_type"]

hf_datasets_serializers: SerializerTypeList = []
hf_datasets_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("datasets") is not None:
    from pathlib import Path

    class HFDatasetSerializer(Serializer[Any]):
        """Serialize HuggingFace ``Dataset`` via Arrow-based ``save_to_disk``."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            from datasets import Dataset

            return isinstance(obj, Dataset)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import datasets

            obj.save_to_disk(str(directory / "dataset"))
            write_meta(directory, HFDatasetSerializer, datasets_version=datasets.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            from datasets import Dataset

            return Dataset.load_from_disk(str(directory / "dataset"))

    class HFDatasetDictSerializer(Serializer[Any]):
        """Serialize HuggingFace ``DatasetDict`` via Arrow-based ``save_to_disk``."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            from datasets import DatasetDict

            return isinstance(obj, DatasetDict)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import datasets

            obj.save_to_disk(str(directory / "dataset_dict"))
            write_meta(directory, HFDatasetDictSerializer, datasets_version=datasets.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            from datasets import DatasetDict

            return DatasetDict.load_from_disk(str(directory / "dataset_dict"))

    hf_datasets_serializers = [HFDatasetDictSerializer, HFDatasetSerializer]
    hf_datasets_serializers_by_type = {
        "datasets.dataset_dict.DatasetDict": HFDatasetDictSerializer,
        "datasets.arrow_dataset.Dataset": HFDatasetSerializer,
    }
