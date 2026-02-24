"""Handlers for Hugging Face datasets objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry

__all__ = ["datasets_handlers", "datasets_handlers_by_type"]


datasets_handlers: HandlerTypeList = []
datasets_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("datasets") is not None:

    def _has_datasets_base(obj: Any, type_name: str) -> bool:
        return any(
            base.__name__ == type_name and base.__module__.split(".")[0] == "datasets"
            for base in type(obj).__mro__
        )

    def _features_payload(obj: Any) -> Any:
        features = getattr(obj, "features", None)
        if features is None:
            return None
        if hasattr(features, "to_dict"):
            return features.to_dict()
        return str(features)

    class HFDatasetHandler(CollectionHandler):
        """Hash Hugging Face Dataset objects by fingerprint and schema metadata."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_datasets_base(obj, "Dataset") or _has_datasets_base(obj, "IterableDataset")

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            return [
                getattr(obj, "_fingerprint", None),
                getattr(obj, "num_rows", None),
                getattr(obj, "column_names", None),
                str(getattr(obj, "split", None)),
                _features_payload(obj),
            ]

    class HFDatasetDictHandler(CollectionHandler):
        """Hash Hugging Face DatasetDict objects by sorted split entries."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_datasets_base(obj, "DatasetDict") or _has_datasets_base(obj, "IterableDatasetDict")

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            keys = sorted(obj.keys())
            return [(key, obj[key]) for key in keys]

    datasets_handlers = [HFDatasetHandler, HFDatasetDictHandler]
    datasets_handlers_by_type = {
        "datasets.arrow_dataset.Dataset": HFDatasetHandler,
        "datasets.iterable_dataset.IterableDataset": HFDatasetHandler,
        "datasets.dataset_dict.DatasetDict": HFDatasetDictHandler,
        "datasets.dataset_dict.IterableDatasetDict": HFDatasetDictHandler,
    }
