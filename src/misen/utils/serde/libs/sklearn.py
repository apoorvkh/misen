"""Serializer for scikit-learn estimators via joblib."""

import importlib.util
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import (
    Serializer,
    SerializerTypeRegistry,
)

__all__ = ["sklearn_serializers", "sklearn_serializers_by_type"]

sklearn_serializers: list[type[Serializer]] = []
sklearn_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("sklearn") is not None and importlib.util.find_spec("joblib") is not None:
    from pathlib import Path

    class SklearnEstimatorSerializer(Serializer[Any]):
        """Serialize scikit-learn estimators via ``joblib``.

        Note: joblib uses pickle internally. Serialized models may not load
        across different scikit-learn versions. The sklearn version is recorded
        in metadata for diagnostics.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            from sklearn.base import BaseEstimator

            return isinstance(obj, BaseEstimator)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import joblib
            import sklearn

            joblib.dump(obj, directory / "model.joblib")
            return {
                "sklearn_version": sklearn.__version__,
                "joblib_version": joblib.__version__,
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import joblib

            return joblib.load(directory / "model.joblib")

    sklearn_serializers = [SklearnEstimatorSerializer]
    sklearn_serializers_by_type = {"sklearn.base.BaseEstimator": SklearnEstimatorSerializer}
