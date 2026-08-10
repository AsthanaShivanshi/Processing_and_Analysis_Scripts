from __future__ import annotations

import numpy as np
import xarray as xr

EPS = 1e-30

import matplotlib.pyplot as plt


def _radial_average(array_2d: np.ndarray) -> np.ndarray:
    y, x = np.indices(array_2d.shape)
    center_x = (array_2d.shape[1] - 1) / 2.0
    center_y = (array_2d.shape[0] - 1) / 2.0
    r = np.hypot(x - center_x, y - center_y).astype(np.int32)

    tbin = np.bincount(r.ravel(), array_2d.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def _field_profile(field_2d: np.ndarray) -> np.ndarray:
    field_2d = np.asarray(field_2d, dtype=np.float64)
    field_2d = np.nan_to_num(field_2d, nan=0.0)

    if field_2d.ndim != 2:
        raise ValueError(f"_field_profile expects a 2D field, got shape {field_2d.shape}")

    fft_field = np.fft.fftshift(np.fft.fft2(field_2d), axes=(-2, -1))
    power = np.abs(fft_field) ** 2
    return _radial_average(power)


def _average_profiles(profiles: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not profiles:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    min_len = min(p.size for p in profiles)
    if min_len == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    trimmed = [p[:min_len] for p in profiles]
    ps_mean = np.mean(np.stack(trimmed, axis=0), axis=0)
    k = np.arange(min_len, dtype=np.float64)
    return k, ps_mean


def _to_realizations(da: xr.DataArray) -> np.ndarray:
    """
    Convert a DataArray to an array shaped (n, ny, nx), where n is the number
    of realizations formed by flattening all leading non-spatial dimensions.
    """
    arr = np.asarray(np.nan_to_num(da.values, nan=0.0), dtype=np.float64)
    arr = np.squeeze(arr)

    if arr.ndim < 2:
        raise ValueError("RAPSD requires at least two spatial dimensions.")

    if arr.ndim == 2:
        return arr[None, :, :]

    return arr.reshape(-1, arr.shape[-2], arr.shape[-1])


def _profiles_from_dataarray(da: xr.DataArray) -> list[np.ndarray]:
    arr = _to_realizations(da)
    return [_field_profile(arr[i]) for i in range(arr.shape[0])]


def rapsd(
    da: xr.DataArray,
    time_dim: str | None = None,
    sample_dim: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean Radially Averaged Power Spectral Density (RAPSD).

    Behavior:
    - time_dim is None:
        Compute RAPSD over all realizations by flattening leading dimensions.
    - time_dim is provided:
        Group by time, compute a RAPSD for each time slice, then average over time.
    - sample_dim is provided:
        Within each time slice, compute RAPSD for each sample/member, average
        over members, then average over time.

    For ensemble means, call this on the ensemble-mean DataArray with no
    sample_dim.
    """
    if time_dim is None:
        profiles = _profiles_from_dataarray(da)
        k, ps_mean = _average_profiles(profiles)
        keep = (k > 0) & np.isfinite(ps_mean) & (ps_mean > 0)
        return k[keep], ps_mean[keep]

    if time_dim not in da.dims:
        raise ValueError(f"time_dim='{time_dim}' not found in DataArray dims={da.dims}")

    if sample_dim is not None and sample_dim not in da.dims:
        raise ValueError(
            f"sample_dim='{sample_dim}' not found in DataArray dims={da.dims}"
        )

    time_profiles: list[np.ndarray] = []

    for _, da_t in da.groupby(time_dim):
        da_t = da_t.squeeze(drop=True)

        if sample_dim is not None and sample_dim in da_t.dims:
            sample_profiles: list[np.ndarray] = []
            for s in da_t[sample_dim].values:
                field_da = da_t.sel({sample_dim: s}).squeeze(drop=True)
                sample_profiles.extend(_profiles_from_dataarray(field_da))

            _, ps_t = _average_profiles(sample_profiles)
            if ps_t.size:
                time_profiles.append(ps_t)
        else:
            profiles = _profiles_from_dataarray(da_t)
            _, ps_t = _average_profiles(profiles)
            if ps_t.size:
                time_profiles.append(ps_t)

    k, ps_mean = _average_profiles(time_profiles)
    keep = (k > 0) & np.isfinite(ps_mean) & (ps_mean > 0)
    return k[keep], ps_mean[keep]


def spectral_ratio(
    da_pred: xr.DataArray,
    da_obs: xr.DataArray,
    time_dim: str | None = None,
    sample_dim: str | None = None,
    label_pred: str = "pred",
    label_obs: str = "obs",
    ax=None,
):
    """
    Plot spectral ratio (pred/obs) vs wavenumber k using rapsd().
    """
    k_pred, ps_pred = rapsd(da_pred, time_dim=time_dim, sample_dim=sample_dim)
    k_obs, ps_obs = rapsd(da_obs, time_dim=time_dim, sample_dim=None)

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
    ax.axhline(1.0, color="k", lw=1, ls="--", label="perfect")
    ax.set_xscale("log")
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel(f"Spectral ratio ({label_pred} / {label_obs})")
    ax.set_title("Temporally averaged spectral ratio (pred/obs)")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    return ax


def ralsd(
    k_ref: np.ndarray,
    ps_ref: np.ndarray,
    k_mod: np.ndarray,
    ps_mod: np.ndarray,
) -> float:
    """
    Radially Averaged Log Spectral Distance.
    """
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