"""Serializer for scikit-learn estimators via joblib."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["sklearn_serializers", "sklearn_serializers_by_type"]

sklearn_serializers: SerializerTypeList = []
sklearn_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("sklearn") is not None and importlib.util.find_spec("joblib") is not None:
    from pathlib import Path

    class SklearnEstimatorSerializer(Serializer[Any]):
        """Serialize scikit-learn estimators via ``joblib``.

        Note: joblib uses pickle internally. Serialized models may not load
        across different scikit-learn versions. The sklearn version is recorded
        in metadata for diagnostics.
        """

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            from sklearn.base import BaseEstimator

            return isinstance(obj, BaseEstimator)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import joblib
            import sklearn

            joblib.dump(obj, directory / "model.joblib")
            write_meta(
                directory,
                SklearnEstimatorSerializer,
                sklearn_version=sklearn.__version__,
                joblib_version=joblib.__version__,
            )

        @staticmethod
        def load(directory: Path) -> Any:
            import joblib

            return joblib.load(directory / "model.joblib")

    sklearn_serializers = [SklearnEstimatorSerializer]
    sklearn_serializers_by_type = {"sklearn.base.BaseEstimator": SklearnEstimatorSerializer}
