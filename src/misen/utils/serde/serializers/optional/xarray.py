"""Serializers for xarray Dataset and DataArray via NetCDF."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["xarray_serializers", "xarray_serializers_by_type"]

xarray_serializers: SerializerTypeList = []
xarray_serializers_by_type: SerializerTypeRegistry = {}

# Require xarray and at least one NetCDF engine.
if importlib.util.find_spec("xarray") is not None and (
    importlib.util.find_spec("netCDF4") is not None
    or importlib.util.find_spec("h5netcdf") is not None
    or importlib.util.find_spec("scipy") is not None
):
    from pathlib import Path

    class XarrayDatasetSerializer(Serializer[Any]):
        """Serialize ``xarray.Dataset`` via NetCDF4."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import xarray as xr

            return isinstance(obj, xr.Dataset)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import xarray as xr

            obj.to_netcdf(directory / "data.nc")
            write_meta(directory, XarrayDatasetSerializer, xarray_version=xr.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            import xarray as xr

            ds = xr.open_dataset(directory / "data.nc")
            ds.load()
            ds.close()
            return ds

    class XarrayDataArraySerializer(Serializer[Any]):
        """Serialize ``xarray.DataArray`` via NetCDF4."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            import xarray as xr

            return isinstance(obj, xr.DataArray)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import xarray as xr

            obj.to_netcdf(directory / "data.nc")
            write_meta(directory, XarrayDataArraySerializer, xarray_version=xr.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            import xarray as xr

            da = xr.open_dataarray(directory / "data.nc")
            da.load()
            da.close()
            return da

    xarray_serializers = [XarrayDatasetSerializer, XarrayDataArraySerializer]
    xarray_serializers_by_type = {
        "xarray.core.dataset.Dataset": XarrayDatasetSerializer,
        "xarray.core.dataarray.DataArray": XarrayDataArraySerializer,
    }
