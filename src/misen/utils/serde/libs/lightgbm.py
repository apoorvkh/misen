"""Serializer for LightGBM ``Booster`` via the native text format.

Only ``lightgbm.Booster`` round-trips here.  The sklearn-API wrappers
(``LGBMRegressor``, ``LGBMClassifier``, ``LGBMRanker``) carry their
``__init__`` hyperparameters (``learning_rate``, ``n_estimators``, ...)
on the Python wrapper object — not in ``Booster.save_model``'s text
file — so a wrapper round-trip would silently restore those values as
defaults instead of the trained ones.  We match the wrappers too only
to surface that limitation as an explicit error pointing users to
``model.booster_``.
"""

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.exceptions import SerializationError
from misen.utils.serde.base import Serializer

__all__ = ["lightgbm_serializers", "lightgbm_serializers_by_type"]

lightgbm_serializers: list[type[Serializer]] = []
lightgbm_serializers_by_type: dict[str, type[Serializer]] = {}

if importlib.util.find_spec("lightgbm") is not None:

    class LightGBMBoosterSerializer(Serializer[Any]):
        """Serialize ``lightgbm.Booster`` via the native text format."""

        @staticmethod
        def match(obj: Any) -> bool:
            import lightgbm as lgb

            return isinstance(obj, (lgb.Booster, lgb.LGBMModel))

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import lightgbm as lgb

            if isinstance(obj, lgb.LGBMModel):
                msg = (
                    f"{type(obj).__name__} cannot round-trip faithfully — its sklearn-API "
                    "``__init__`` hyperparameters (learning_rate, n_estimators, ...) live on "
                    "the wrapper, not in ``Booster.save_model``'s text file, so they would "
                    "silently restore as defaults.  Save ``model.booster_`` (a "
                    "``lightgbm.Booster``) instead."
                )
                raise SerializationError(msg)
            obj.save_model(str(directory / "model.txt"))
            return {"lightgbm_version": lgb.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import lightgbm as lgb

            return lgb.Booster(model_file=str(directory / "model.txt"))

    lightgbm_serializers = [LightGBMBoosterSerializer]
    lightgbm_serializers_by_type = {
        "lightgbm.basic.Booster": LightGBMBoosterSerializer,
    }
