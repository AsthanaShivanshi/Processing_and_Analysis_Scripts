from __future__ import annotations

import numpy as np
import xarray as xr

SAMPLE_DIM_CANDIDATES = ("member", "sample", "samples")


def _resolve_sample_dim(da: xr.DataArray, sample_dim: str | None = "auto") -> str | None:
    if sample_dim is None:
        return None

    if sample_dim != "auto":
        if sample_dim not in da.dims:
            raise ValueError(f"sample dimension '{sample_dim}' not found in dims={da.dims}")
        return sample_dim

    for d in SAMPLE_DIM_CANDIDATES:
        if d in da.dims:
            return d
    return None


def _single_sample_rmse(pred: xr.DataArray, ref: xr.DataArray) -> float:
    pred, ref = xr.align(pred, ref, join="inner")
    if pred.size == 0 or ref.size == 0:
        return np.nan

    diff2 = (pred - ref) ** 2
    return float(np.sqrt(diff2.mean(skipna=True).values))


def rmse_gridwise_spatial_mean(
    pred: xr.DataArray,
    ref: xr.DataArray,
    sample_dim: str = "auto",
    mode: str = "per_sample_mean",
) -> float:
    """
    Standard RMSE.

    Behavior:
    - If no sample/member dimension exists:
        RMSE = sqrt(mean((pred - ref)^2)) over all paired valid values.
    - If a sample/member dimension exists:
        compute RMSE for each sample/member separately,
        then return the mean of those RMSE values.

    mode is kept for compatibility.
    """
    if mode != "per_sample_mean":
        raise ValueError("Only mode='per_sample_mean' is supported in this configuration.")

    sd = _resolve_sample_dim(pred, sample_dim)

    if sd is None:
        return _single_sample_rmse(pred, ref)

    if sd in ref.dims and ref.sizes[sd] != pred.sizes[sd]:
        raise ValueError(
            f"sample dimension '{sd}' size mismatch: {pred.sizes[sd]} vs {ref.sizes[sd]}"
        )

    vals = []
    for i in range(pred.sizes[sd]):
        pred_i = pred.isel({sd: i}, drop=True)
        ref_i = ref.isel({sd: i}, drop=True) if sd in ref.dims else ref
        vals.append(_single_sample_rmse(pred_i, ref_i))

    arr = np.asarray(vals, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan