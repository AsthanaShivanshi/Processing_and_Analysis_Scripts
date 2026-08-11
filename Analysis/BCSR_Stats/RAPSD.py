from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

EPS = 1e-30
SAMPLE_DIM_CANDIDATES = ("member", "sample", "samples")


def _resolve_sample_dim(da: xr.DataArray, sample_dim: str | None = "auto") -> str | None:
    if sample_dim is None:
        return None
    if sample_dim != "auto":
        if sample_dim not in da.dims:
            raise ValueError(f"sample_dim='{sample_dim}' not found in dims={da.dims}")
        return sample_dim
    for d in SAMPLE_DIM_CANDIDATES:
        if d in da.dims:
            return d
    return None


def _radial_average(array_2d: np.ndarray) -> np.ndarray:
    y, x = np.indices(array_2d.shape)
    cx = (array_2d.shape[1] - 1) / 2.0
    cy = (array_2d.shape[0] - 1) / 2.0
    r = np.hypot(x - cx, y - cy).astype(np.int32)

    tbin = np.bincount(r.ravel(), weights=array_2d.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def _field_profile(field_2d: np.ndarray, normalize: bool = False) -> np.ndarray:
    field_2d = np.asarray(field_2d, dtype=np.float64)
    field_2d = np.nan_to_num(field_2d, nan=0.0)

    if field_2d.ndim != 2:
        raise ValueError(f"_field_profile expects 2D input, got shape {field_2d.shape}")

    fft_field = np.fft.fftshift(np.fft.fft2(field_2d), axes=(-2, -1))
    power = np.abs(fft_field) ** 2
    profile = _radial_average(power)

    if normalize:
        s = np.sum(profile)
        if s > 0:
            profile = profile / s

    return profile


def _iter_realization_fields(
    da: xr.DataArray,
    time_dim: str | None = None,
    sample_dim: str | None = None,
):
    real_dims: list[str] = []

    if time_dim is not None and time_dim in da.dims:
        real_dims.append(time_dim)

    sd = _resolve_sample_dim(da, sample_dim)
    if sd is not None and sd not in real_dims:
        real_dims.append(sd)

    spatial_dims = [d for d in da.dims if d not in real_dims]
    if len(spatial_dims) != 2:
        raise ValueError(
            f"RAPSD expects exactly 2 spatial dims after removing realization dims; got {spatial_dims}"
        )

    if not real_dims:
        field = da.transpose(*spatial_dims)
        vals = np.asarray(field.values, dtype=np.float64)
        vals = np.nan_to_num(vals, nan=0.0)
        if vals.ndim != 2:
            raise ValueError(f"RAPSD requires 2D spatial input, got shape {vals.shape}")
        yield vals
        return

    shape = [da.sizes[d] for d in real_dims]
    for idx in np.ndindex(*shape):
        indexers = {d: i for d, i in zip(real_dims, idx)}
        field = da.isel(indexers).transpose(*spatial_dims)
        vals = np.asarray(field.values, dtype=np.float64)
        vals = np.nan_to_num(vals, nan=0.0)
        if vals.ndim != 2:
            raise ValueError(f"RAPSD requires 2D spatial input, got shape {vals.shape}")
        yield vals


def _average_profiles_streaming(
    profiles,
) -> tuple[np.ndarray, np.ndarray]:
    sum_prof: np.ndarray | None = None
    min_len: int | None = None
    n = 0

    for prof in profiles:
        prof = np.asarray(prof, dtype=np.float64)
        if prof.size == 0:
            continue

        if sum_prof is None:
            sum_prof = prof.copy()
            min_len = prof.size
            n = 1
            continue

        new_min = min(min_len, prof.size)
        if new_min < min_len:
            sum_prof = sum_prof[:new_min]
            min_len = new_min

        sum_prof[:min_len] += prof[:min_len]
        n += 1

    if sum_prof is None or min_len is None or n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    ps_mean = sum_prof[:min_len] / n
    k = np.arange(min_len, dtype=np.float64)
    return k, ps_mean


def rapsd(
    da: xr.DataArray,
    time_dim: str | None = None,
    sample_dim: str | None = None,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean RAPSD over all realizations.

    For unpaired/distributional evaluation:
    - pass time_dim="time"
    - pass sample_dim for ensembles
    - compare pooled spectra with ralsd()
    """
    def _profiles():
        for field in _iter_realization_fields(da, time_dim=time_dim, sample_dim=sample_dim):
            yield _field_profile(field, normalize=normalize)

    k, ps_mean = _average_profiles_streaming(_profiles())
    keep = (k > 0) & np.isfinite(ps_mean) & (ps_mean > 0)
    return k[keep], ps_mean[keep]


def ralsd(
    k_ref: np.ndarray,
    ps_ref: np.ndarray,
    k_mod: np.ndarray,
    ps_mod: np.ndarray,
) -> float:
    if k_ref.size == 0 or k_mod.size == 0:
        return np.nan

    n = min(k_ref.size, k_mod.size)
    p_ref = np.asarray(ps_ref[:n], dtype=np.float64)
    p_mod = np.asarray(ps_mod[:n], dtype=np.float64)

    mask = np.isfinite(p_ref) & np.isfinite(p_mod) & (p_ref > 0) & (p_mod > 0)
    if not np.any(mask):
        return np.nan

    ratio = p_ref[mask] / np.maximum(p_mod[mask], EPS)
    log_ratio = 10.0 * np.log10(np.maximum(ratio, EPS))
    return float(np.mean(log_ratio**2))


def spectral_ratio(
    da_pred: xr.DataArray,
    da_obs: xr.DataArray,
    time_dim: str | None = None,
    sample_dim: str | None = None,
    normalize: bool = False,
    ax=None,
):
    k_pred, ps_pred = rapsd(da_pred, time_dim=time_dim, sample_dim=sample_dim, normalize=normalize)
    k_obs, ps_obs = rapsd(da_obs, time_dim=time_dim, sample_dim=None, normalize=normalize)

    n = min(k_pred.size, k_obs.size)
    if n == 0:
        raise ValueError("Empty spectra returned by rapsd().")

    k_grid = k_pred[:n]
    ps_pred_i = np.interp(k_grid, k_pred, ps_pred)
    ps_obs_i = np.interp(k_grid, k_obs, ps_obs)

    ratio = ps_pred_i / np.maximum(ps_obs_i, EPS)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.plot(k_grid, ratio, lw=2, color="steelblue")
    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Spectral ratio (pred / obs)")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    return ax