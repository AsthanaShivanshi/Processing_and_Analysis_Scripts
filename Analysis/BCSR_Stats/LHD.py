from __future__ import annotations

import numpy as np
import xarray as xr

from rmse import _daily_climatology


def _to_1d_array(x) -> np.ndarray:
    if isinstance(x, xr.DataArray):
        x = x.values
    arr = np.asarray(x)
    return arr.ravel()


def lhd(pred, ref, bins=50, min_count=10, eps=1e-12) -> float:
    p = _to_1d_array(pred)
    r = _to_1d_array(ref)

    m = np.isfinite(p) & np.isfinite(r)
    p, r = p[m], r[m]

    if p.size == 0 or r.size == 0:
        return np.nan

    vmin, vmax = np.nanmin(np.r_[p, r]), np.nanmax(np.r_[p, r])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.nan

    edges = np.linspace(vmin, vmax, bins + 1)
    hp, _ = np.histogram(p, bins=edges)
    hr, _ = np.histogram(r, bins=edges)

    valid = (hp > min_count) & (hr > min_count)
    if not np.any(valid):
        return np.nan

    hp = hp[valid] / hp[valid].sum()
    hr = hr[valid] / hr[valid].sum()
    return float(np.sqrt(np.mean((np.log(hp + eps) - np.log(hr + eps)) ** 2)))


def lhd_gridwise_spatial_mean(
    pred: xr.DataArray,
    ref: xr.DataArray,
    bins=50,
    min_count=10,
    eps=1e-12,
) -> float:
    pred, ref = xr.align(pred, ref, join="inner")
    if pred.sizes.get("time", 0) == 0:
        return np.nan

    clim_pred = _daily_climatology(pred)
    clim_ref = _daily_climatology(ref)

    spatial_dims = [d for d in clim_pred.dims if d != "dayofyear"]

    def _lhd_1d(p, r) -> float:
        return lhd(p, r, bins=bins, min_count=min_count, eps=eps)

    lhd_map = xr.apply_ufunc(
        _lhd_1d,
        clim_pred,
        clim_ref,
        input_core_dims=[["dayofyear"], ["dayofyear"]],
        output_dtypes=[float],
        vectorize=True,
    )

    if not spatial_dims:
        return float(lhd_map.values)

    return float(lhd_map.mean(dim=spatial_dims, skipna=True).values)