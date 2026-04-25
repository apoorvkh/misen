"""Serializer for arviz ``InferenceData`` via NetCDF.

:class:`arviz.InferenceData` is the canonical container for Bayesian
posterior/prior/observed traces across PyMC, NumPyro, Stan, and
CmdStanPy.  The library ships ``to_netcdf`` / ``from_netcdf`` which
preserves all groups, chain/draw dims, and xarray-level metadata.
"""

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.utils.serde.base import Serializer

__all__ = ["arviz_serializers", "arviz_serializers_by_type"]

arviz_serializers: list[type[Serializer]] = []
arviz_serializers_by_type: dict[str, type[Serializer]] = {}


# ``InferenceData.to_netcdf`` needs a NetCDF engine (netCDF4 or h5netcdf);
# without one the serializer would fail at write time, so guard it here.
if importlib.util.find_spec("arviz") is not None and (
    importlib.util.find_spec("netCDF4") is not None or importlib.util.find_spec("h5netcdf") is not None
):

    class ArvizInferenceDataSerializer(Serializer[Any]):
        """Serialize ``arviz.InferenceData`` via NetCDF."""

        @staticmethod
        def match(obj: Any) -> bool:
            import arviz as az

            return isinstance(obj, az.InferenceData)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import arviz as az

            obj.to_netcdf(str(directory / "data.nc"))
            return {"arviz_version": az.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import arviz as az

            return az.from_netcdf(str(directory / "data.nc"))

    arviz_serializers = [ArvizInferenceDataSerializer]
    arviz_serializers_by_type = {
        "arviz.data.inference_data.InferenceData": ArvizInferenceDataSerializer,
    }
