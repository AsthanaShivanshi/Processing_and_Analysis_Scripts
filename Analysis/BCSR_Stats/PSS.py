from __future__ import annotations

import numpy as np
import xarray as xr

SAMPLE_DIM_CANDIDATES = ("member", "sample")


def daily_climatology(arr, time_coord=None):
    if isinstance(arr, xr.DataArray):
        if time_coord is None:
            time_coord = arr["time"]
        values = arr.values
    else:
        values = np.asarray(arr)

    if time_coord is None:
        raise ValueError("time_coord is required")

    if hasattr(time_coord, "dt"):
        dayofyear = np.asarray(time_coord.dt.dayofyear)
    else:
        dayofyear = np.asarray(time_coord)

    clim = np.full((366, *values.shape[1:]), np.nan)
    for doy in range(1, 367):
        mask = dayofyear == doy
        if np.any(mask):
            clim[doy - 1] = np.nanmean(values[mask], axis=0)
        else:
            clim[doy - 1] = np.nan
    return clim


def gridwise_perkins_skill_score_unpaired(a, b, nbins=50, min_count=10):
    """
    Perkins Skill Score on grid, allowing different sample lengths:
      a.shape = (n_a, ...spatial...)
      b.shape = (n_b, ...spatial...)
    """
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

        try:
            combined = np.concatenate([a_valid, b_valid])
            vmin = np.nanmin(combined)
            vmax = np.nanmax(combined)
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-6

            bins = np.linspace(vmin, vmax, nbins + 1)
            hist_a, _ = np.histogram(a_valid, bins=bins, density=True)
            hist_b, _ = np.histogram(b_valid, bins=bins, density=True)

            sa = hist_a.sum()
            sb = hist_b.sum()
            if sa > 0 and sb > 0:
                hist_a = hist_a / sa
                hist_b = hist_b / sb
                pss[idx] = np.minimum(hist_a, hist_b).sum()
        except Exception:
            pss[idx] = np.nan

    return pss


def _spatial_dims(da):
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


def _to_time_first_values(da: xr.DataArray, sample_dim: str | None, pool_sample: bool) -> np.ndarray:
    if "time" not in da.dims:
        raise ValueError("DataArray must have 'time' dimension")

    dims = list(da.dims)
    t_ax = dims.index("time")

    if pool_sample and sample_dim is not None and sample_dim in da.dims:
        s_ax = dims.index(sample_dim)
        order = [t_ax, s_ax] + [i for i in range(da.ndim) if i not in (t_ax, s_ax)]
        vals = np.transpose(da.values, order)
        nt, ns = vals.shape[:2]
        return vals.reshape(nt * ns, *vals.shape[2:])

    order = [t_ax] + [i for i in range(da.ndim) if i != t_ax]
    return np.transpose(da.values, order)


def _single_sample_pss_gridwise_spatial_mean(pred, ref, nbins=50):
    pred, ref = xr.align(pred, ref, join="inner")
    if pred.sizes.get("time", 0) == 0:
        return np.nan

    clim_a = daily_climatology(pred, pred["time"])
    clim_b = daily_climatology(ref, ref["time"])

    pss_map = gridwise_perkins_skill_score_unpaired(clim_a, clim_b, nbins=nbins)
    return float(np.nanmean(pss_map))


def _pooled_pss_gridwise_spatial_mean(pred, ref, nbins=50, sample_dim="auto"):
    pred, ref = xr.align(pred, ref, join="inner")
    if pred.sizes.get("time", 0) == 0 or ref.sizes.get("time", 0) == 0:
        return np.nan

    sd_pred = _resolve_sample_dim(pred, sample_dim)
    sd_ref = _resolve_sample_dim(ref, "auto")

    # Pool generated draws over (time, sample); obs stays over time (unless it also has sample, then pool it too)
    a = _to_time_first_values(pred, sample_dim=sd_pred, pool_sample=(sd_pred is not None))
    b = _to_time_first_values(ref, sample_dim=sd_ref, pool_sample=(sd_ref is not None))

    pss_map = gridwise_perkins_skill_score_unpaired(a, b, nbins=nbins)
    return float(np.nanmean(pss_map))


def pss_gridwise_spatial_mean(pred, ref, nbins=50, sample_dim="auto", mode="pooled"):
    """
    mode:
      - 'pooled'          : pool pred draws over (time,sample), compare to ref over time
      - 'per_sample_mean' : old behavior (mean of per-sample scores using daily climatology)
    """
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