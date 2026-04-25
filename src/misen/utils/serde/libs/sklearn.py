"""sklearn estimators are intentionally unsupported.

Two faithful round-trip paths exist on paper, but neither satisfies
both fidelity and version-stability:

- ``joblib.dump`` round-trips the estimator faithfully (same class,
  same fitted attributes, same ``.fit()`` capability), but sklearn
  itself documents that joblib-pickled estimators are not portable
  across sklearn versions.
- ``skl2onnx`` produces a version-stable artifact, but the loaded
  object is an ONNX inference session — a different Python type
  with no ``.fit()``, no ``.coef_``, and only the inference subset
  of the sklearn API.  Round-trip is not faithful.

Until sklearn ships a stable, lossless persistence path, users
should serialize the *inputs* needed to refit the estimator (the
training data, the hyperparameters) and refit on load.
"""

from misen.utils.serde.base import Serializer

__all__ = ["sklearn_serializers", "sklearn_serializers_by_type"]

sklearn_serializers: list[type[Serializer]] = []
sklearn_serializers_by_type: dict[str, type[Serializer]] = {}
