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


def mbe_gridwise_spatial_mean(
    pred: xr.DataArray,
    ref: xr.DataArray,
    sample_dim: str = "auto",
) -> float:
    sd = _resolve_sample_dim(pred, sample_dim)

    if sd is not None:
        pred = pred.mean(dim=sd, skipna=True)
    if sd is not None and sd in ref.dims:
        ref = ref.mean(dim=sd, skipna=True)

    pred, ref = xr.align(pred, ref, join="inner")
    if pred.size == 0 or ref.size == 0:
        return np.nan

    bias = pred - ref

    if "time" in bias.dims:
        bias = bias.mean(dim="time", skipna=True)

    spatial_dims = [d for d in bias.dims if d != "time"]
    if spatial_dims:
        bias = bias.mean(dim=spatial_dims, skipna=True)

    return float(bias.values)