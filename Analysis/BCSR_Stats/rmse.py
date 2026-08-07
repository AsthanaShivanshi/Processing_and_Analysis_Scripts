from __future__ import annotations

import numpy as np
import xarray as xr

SAMPLE_DIM_CANDIDATES = ("member", "sample")


def _spatial_dims(da: xr.DataArray) -> list[str]:
    return [d for d in da.dims if d != "time"]


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
    if pred.size == 0 or ref.size == 0:
        return np.nan

    diff2 = (pred - ref) ** 2

    if "time" in diff2.dims:
        rmse_map = np.sqrt(diff2.mean(dim="time", skipna=True))
    else:
        rmse_map = np.sqrt(diff2)

    dims = _spatial_dims(rmse_map)
    if not dims:
        return float(rmse_map.values)

    return float(rmse_map.mean(dim=dims, skipna=True).values)


def rmse_gridwise_spatial_mean(
    pred: xr.DataArray,
    ref: xr.DataArray,
    sample_dim: str = "auto",
    mode: str = "per_sample_mean",
) -> float:
    """
    Compute RMSE gridwise over time, then spatially average.

    For pooled data:
    - compute RMSE separately for each sample/member
    - average the resulting scalar RMSEs
    """

    
    if mode != "per_sample_mean":
        raise ValueError("Only mode='per_sample_mean' is supported in this configuration.")

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