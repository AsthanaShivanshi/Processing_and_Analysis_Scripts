from __future__ import annotations

import numpy as np
import xarray as xr

EPS = 1e-30


def _radial_average(array_2d: np.ndarray) -> np.ndarray:
    y, x = np.indices(array_2d.shape)
    center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1]).astype(np.int32)
    tbin = np.bincount(r.ravel(), array_2d.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def _field_profile(field_2d: np.ndarray) -> np.ndarray:
    fft_field = np.fft.fftshift(np.fft.fft2(field_2d, axes=(-2, -1)), axes=(-2, -1))
    power = np.abs(fft_field) ** 2
    return _radial_average(power)


def _average_profiles(profiles: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not profiles:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    min_len = min(p.size for p in profiles)
    profiles = [p[:min_len] for p in profiles]
    ps_mean = np.mean(np.stack(profiles, axis=0), axis=0)
    k = np.arange(min_len, dtype=np.float64)
    return k, ps_mean


def _to_realizations(da: xr.DataArray) -> np.ndarray:
    """
    Convert a DataArray to an array shaped (n, ny, nx), where n is the number
    of realizations formed by flattening all leading non-spatial dimensions.
    """
    arr = np.asarray(np.nan_to_num(da.values, nan=0.0), dtype=np.float64)

    if arr.ndim < 2:
        raise ValueError("RAPSD requires at least two spatial dimensions.")

    if arr.ndim == 2:
        return arr[None, :, :]

    return arr.reshape(-1, arr.shape[-2], arr.shape[-1])


def rapsd(
    da: xr.DataArray,
    time_dim: str | None = None,
    sample_dim: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean Radially Averaged Power Spectral Density (RAPSD).
    """
    if time_dim is None:
        arr = _to_realizations(da)

        profiles = []
        for i in range(arr.shape[0]):
            profiles.append(_field_profile(arr[i]))

        k, ps_mean = _average_profiles(profiles)
        keep = (k > 0) & np.isfinite(ps_mean) & (ps_mean > 0)
        return k[keep], ps_mean[keep]

    if time_dim not in da.dims:
        raise ValueError(f"time_dim='{time_dim}' not found in DataArray dims.")

    if sample_dim is not None and sample_dim not in da.dims:
        raise ValueError(f"sample_dim='{sample_dim}' not found in DataArray dims.")

    time_profiles: list[np.ndarray] = []

    for _, da_t in da.groupby(time_dim):
        da_t = da_t.squeeze(drop=True)

        if sample_dim is not None and sample_dim in da_t.dims:
            sample_profiles = []
            for s in da_t[sample_dim].values:
                field_da = da_t.sel({sample_dim: s}).squeeze(drop=True)
                field = np.asarray(np.nan_to_num(field_da.values, nan=0.0), dtype=np.float64)
                field = np.squeeze(field)

                if field.ndim != 2:
                    raise ValueError(
                        "Each timestep/sample selection must reduce to a 2D field. "
                        f"Got shape {field.shape} — check that lat/lon are the only "
                        "remaining dims after selecting time and sample."
                    )
                sample_profiles.append(_field_profile(field))

            _, ps_t = _average_profiles(sample_profiles)
            time_profiles.append(ps_t)
        else:
            field = np.asarray(np.nan_to_num(da_t.values, nan=0.0), dtype=np.float64)
            field = np.squeeze(field)

            if field.ndim == 2:
                time_profiles.append(_field_profile(field))
            else:
                arr_t = field.reshape(-1, field.shape[-2], field.shape[-1])
                sub_profiles = [_field_profile(arr_t[i]) for i in range(arr_t.shape[0])]
                _, ps_t = _average_profiles(sub_profiles)
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
    import matplotlib.pyplot as plt

    k_pred, ps_pred = rapsd(da_pred, time_dim=time_dim, sample_dim=sample_dim)
    k_obs, ps_obs = rapsd(da_obs, time_dim=time_dim, sample_dim=None)

    n = min(k_pred.size, k_obs.size)
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