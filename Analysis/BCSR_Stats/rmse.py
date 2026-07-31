from __future__ import annotations

import numpy as np
import xarray as xr


def _spatial_dims(da: xr.DataArray) -> list[str]:
    return [d for d in da.dims if d != "time"]


def _daily_climatology(da: xr.DataArray) -> xr.DataArray:
    if "time" not in da.dims:
        return da

    clim = da.groupby(da["time"].dt.dayofyear).mean(dim="time")
    return clim.reindex(dayofyear=np.arange(1, 367))


def rmse_gridwise_spatial_mean(pred: xr.DataArray, ref: xr.DataArray) -> float:
    pred, ref = xr.align(pred, ref, join="inner")
    if pred.sizes.get("time", 0) == 0:
        return np.nan

    clim_pred = _daily_climatology(pred)
    clim_ref = _daily_climatology(ref)

    rmse_map = np.sqrt(((clim_pred - clim_ref) ** 2).mean(dim="dayofyear", skipna=True))

    dims = _spatial_dims(rmse_map)
    if not dims:
        return float(rmse_map.values)

    return float(rmse_map.mean(dim=dims, skipna=True).values)