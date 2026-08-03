from __future__ import annotations

import numpy as np
import xarray as xr

SAMPLE_DIM_CANDIDATES = ("member", "sample")


def _spatial_dims(da: xr.DataArray) -> list[str]:
    return [d for d in da.dims if d != "time"]


def _daily_climatology(da: xr.DataArray) -> xr.DataArray:
    if "time" not in da.dims:
        return da

    clim = da.groupby(da["time"].dt.dayofyear).mean(dim="time")
    return clim.reindex(dayofyear=np.arange(1, 367))


def _resolve_sample_dim(da: xr.DataArray, sample_dim: str = "auto") -> str | None:
    if sample_dim != "auto":
        if sample_dim not in da.dims:
            raise ValueError(f"sample dimension '{sample_dim}' not found in dims={da.dims}")
        return sample_dim

    for d in SAMPLE_DIM_CANDIDATES:
        if d in da.dims:
            return d
    return None


def _single_sample_rmse_gridwise_spatial_mean(pred: xr.DataArray, ref: xr.DataArray) -> float:
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


def rmse_gridwise_spatial_mean(pred: xr.DataArray, ref: xr.DataArray, sample_dim: str = "auto") -> float:
    sd = _resolve_sample_dim(pred, sample_dim)

    if sd is None:
        return _single_sample_rmse_gridwise_spatial_mean(pred, ref)

    if sd in ref.dims and ref.sizes[sd] != pred.sizes[sd]:
        raise ValueError(f"sample dimension '{sd}' size mismatch: {pred.sizes[sd]} vs {ref.sizes[sd]}")

    vals = []
    for i in range(pred.sizes[sd]):
        pred_i = pred.isel({sd: i}, drop=True)
        ref_i = ref.isel({sd: i}, drop=True) if sd in ref.dims else ref
        vals.append(_single_sample_rmse_gridwise_spatial_mean(pred_i, ref_i))

    arr = np.asarray(vals, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan