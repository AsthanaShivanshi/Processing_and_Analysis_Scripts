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


def _time_sample_flattened_values(da: xr.DataArray, sample_dim: str | None, pool_sample: bool) -> np.ndarray:
    if "time" not in da.dims:
        raise ValueError("DataArray must have a 'time' dimension")

    dims = list(da.dims)
    t_ax = dims.index("time")

    if pool_sample and sample_dim is not None and sample_dim in da.dims:
        s_ax = dims.index(sample_dim)
        order = [t_ax, s_ax] + [i for i in range(da.ndim) if i not in (t_ax, s_ax)]
        vals = np.transpose(da.values, order)
        n = vals.shape[0] * vals.shape[1]
        return vals.reshape(n, *vals.shape[2:])

    order = [t_ax] + [i for i in range(da.ndim) if i != t_ax]
    vals = np.transpose(da.values, order)
    return vals.reshape(vals.shape[0], *vals.shape[1:])


def gridwise_perkins_skill_score_unpaired(a, b, nbins=50, min_count=5):
    a = np.asarray(a)
    b = np.asarray(b)

    if a.ndim < 2 or b.ndim < 2:
        raise ValueError("Inputs must be at least 2D: (samples, spatial...)")
    if a.shape[1:] != b.shape[1:]:
        raise ValueError(f"Spatial shapes do not match: {a.shape[1:]} vs {b.shape[1:]}")

    pss = np.full(a.shape[1:], np.nan, dtype=float)

    for idx in np.ndindex(a.shape[1:]):
        a1 = a[(slice(None),) + idx]
        b1 = b[(slice(None),) + idx]

        a_valid = a1[np.isfinite(a1)]
        b_valid = b1[np.isfinite(b1)]
        if a_valid.size <= min_count or b_valid.size <= min_count:
            continue

        combined = np.concatenate([a_valid, b_valid])
        vmin = np.nanmin(combined)
        vmax = np.nanmax(combined)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6

        bins = np.linspace(vmin, vmax, nbins + 1)
        hist_a, _ = np.histogram(a_valid, bins=bins)
        hist_b, _ = np.histogram(b_valid, bins=bins)

        sa = hist_a.sum()
        sb = hist_b.sum()
        if sa > 0 and sb > 0:
            hist_a = hist_a / sa
            hist_b = hist_b / sb
            pss[idx] = np.minimum(hist_a, hist_b).sum()

    return pss


def _single_sample_pss_gridwise_spatial_mean(pred, ref, nbins=50):
    pred, ref = xr.align(pred, ref, join="inner")
    if pred.sizes.get("time", 0) == 0 or ref.sizes.get("time", 0) == 0:
        return np.nan

    a = _time_sample_flattened_values(pred, sample_dim=None, pool_sample=False)
    b = _time_sample_flattened_values(ref, sample_dim=None, pool_sample=False)

    pss_map = gridwise_perkins_skill_score_unpaired(a, b, nbins=nbins)
    return float(np.nanmean(pss_map))


def _pooled_pss_gridwise_spatial_mean(pred, ref, nbins=50, sample_dim="auto"):
    pred, ref = xr.align(pred, ref, join="inner")
    if pred.sizes.get("time", 0) == 0 or ref.sizes.get("time", 0) == 0:
        return np.nan

    sd_pred = _resolve_sample_dim(pred, sample_dim)
    sd_ref = _resolve_sample_dim(ref, "auto")

    a = _time_sample_flattened_values(pred, sample_dim=sd_pred, pool_sample=True)
    b = _time_sample_flattened_values(ref, sample_dim=sd_ref, pool_sample=True)

    pss_map = gridwise_perkins_skill_score_unpaired(a, b, nbins=nbins)
    return float(np.nanmean(pss_map))


def pss_gridwise_spatial_mean(pred, ref, nbins=50, sample_dim="auto", mode="pooled"):
    if mode not in {"pooled", "per_sample_mean"}:
        raise ValueError("mode must be one of {'pooled', 'per_sample_mean'}")

    if mode == "pooled":
        return _pooled_pss_gridwise_spatial_mean(pred, ref, nbins=nbins, sample_dim=sample_dim)

    sd = _resolve_sample_dim(pred, sample_dim)
    if sd is None:
        return _single_sample_pss_gridwise_spatial_mean(pred, ref, nbins=nbins)

    if sd in ref.dims and ref.sizes[sd] != pred.sizes[sd]:
        raise ValueError(f"sample dimension '{sd}' size mismatch: {pred.sizes[sd]} vs {ref.sizes[sd]}")

    vals = []
    for i in range(pred.sizes[sd]):
        pred_i = pred.isel({sd: i}, drop=True)
        ref_i = ref.isel({sd: i}, drop=True) if sd in ref.dims else ref
        vals.append(_single_sample_pss_gridwise_spatial_mean(pred_i, ref_i, nbins=nbins))

    arr = np.asarray(vals, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan