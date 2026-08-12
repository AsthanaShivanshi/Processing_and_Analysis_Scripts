from __future__ import annotations

import gc
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm

EPS = 1e-30
SAMPLE_DIM_CANDIDATES = ("member", "sample", "samples")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_sample_dim(da: xr.DataArray, sample_dim: str | None = "auto") -> str | None:
    if sample_dim is None:
        return None
    if sample_dim != "auto":
        if sample_dim not in da.dims:
            raise ValueError(f"sample_dim='{sample_dim}' not found in dims={da.dims}")
        return sample_dim
    return next((d for d in SAMPLE_DIM_CANDIDATES if d in da.dims), None)


@lru_cache(maxsize=16)
def _radial_bins(shape: tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape)
    # DC bin is at shape//2 after fftshift — integer centre (Harris et al.)
    r = np.hypot(x - shape[1] // 2, y - shape[0] // 2).astype(np.int32)
    r.flags.writeable = False
    return r


@lru_cache(maxsize=16)
def _bin_counts(shape: tuple[int, int]) -> np.ndarray:
    nr = np.bincount(_radial_bins(shape).ravel())
    nr.flags.writeable = False
    return nr


def _radial_average(array_2d: np.ndarray) -> np.ndarray:
    r  = _radial_bins(array_2d.shape)
    nr = _bin_counts(array_2d.shape)
    return np.bincount(r.ravel(), weights=array_2d.ravel()) / np.maximum(nr, 1)


def _field_profile(field_2d: np.ndarray, normalize: bool = True) -> np.ndarray:
    f = np.nan_to_num(np.asarray(field_2d, dtype=np.float64), nan=0.0)

    # Harris et al. 2022: subtract mean to suppress DC component
    f = f - f.mean()

    # skip near-constant frames (e.g. dry precip days)
    if np.std(f) < 1e-6:
        return np.array([])

    profile = _radial_average(np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2)

    if normalize:
        s = profile.sum()
        if s > 0:
            profile = profile / s

    return profile


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def rapsd(
    da: xr.DataArray,
    time_dim: str | None = "time",
    sample_dim: str | None = "auto",
    normalize: bool = True,
    desc: str = "RAPSD",
    chunk_size: int = 50,          # frames per chunk — controls OOM
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean Radially Averaged Power Spectral Density.

    Iterates over frames one chunk at a time to avoid loading the full
    array into memory (OOM-safe).

    Parameters
    ----------
    da          : input DataArray  (time [x sample] x lat x lon)
    time_dim    : name of the time dimension
    sample_dim  : name of the ensemble/sample dimension or 'auto'
    normalize   : normalize each spectrum by its total power before averaging
    desc        : tqdm label
    chunk_size  : number of time steps loaded into RAM at once

    Returns
    -------
    k       : wavenumber array (positive integers, k > 0)
    ps_mean : mean radial power spectrum
    """
    sd = _resolve_sample_dim(da, sample_dim)

    # identify spatial dims
    real_dims    = [d for d in [time_dim, sd] if d is not None and d in da.dims]
    spatial_dims = [d for d in da.dims if d not in real_dims]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected 2 spatial dims, got {spatial_dims}")

    # reorder so we iterate over leading dims
    da = da.transpose(*real_dims, *spatial_dims)

    n_time   = da.sizes[time_dim] if time_dim and time_dim in da.dims else 1
    n_sample = da.sizes[sd]       if sd and sd in da.dims             else 1
    n_frames = n_time * n_sample

    profiles: list[np.ndarray] = []
    skipped = 0

    with tqdm(total=n_frames, desc=desc, unit="frame", leave=False) as pbar:
        for t_start in range(0, n_time, chunk_size):
            t_end   = min(t_start + chunk_size, n_time)
            # load only this chunk — avoids full-array materialisation
            chunk   = da.isel({time_dim: slice(t_start, t_end)}).values.astype(np.float32)
            # reshape to (frames, H, W)
            chunk   = chunk.reshape(-1, chunk.shape[-2], chunk.shape[-1])

            for frame in chunk:
                profile = _field_profile(frame.astype(np.float64), normalize)
                if profile.size == 0:
                    skipped += 1
                else:
                    profiles.append(profile)
                pbar.update(1)

            # free chunk memory explicitly
            del chunk
            gc.collect()

    if skipped:
        print(f"[rapsd] skipped {skipped}/{n_frames} near-constant frames")

    if not profiles:
        raise RuntimeError("All frames were skipped — no valid spectra computed.")

    min_len = min(p.size for p in profiles)
    stack   = np.stack([p[:min_len] for p in profiles], axis=0)
    ps_mean = stack.mean(axis=0)
    k       = np.arange(min_len, dtype=np.float64)

    keep = (k > 0) & np.isfinite(ps_mean) & (ps_mean > 0)
    return k[keep], ps_mean[keep]


def ralsd(
    k_ref: np.ndarray,
    ps_ref: np.ndarray,
    k_mod: np.ndarray,
    ps_mod: np.ndarray,
    normalized: bool = True,
) -> float:
    """
    RALSD = sqrt( mean( (10 log10( S_true / S_pred ))^2 ) )
    Harris et al. (2022).

    Both spectra must have been computed with the same normalize flag.
    """
    if k_ref.size == 0 or k_mod.size == 0:
        return np.nan

    n     = min(k_ref.size, k_mod.size)
    p_ref = np.asarray(ps_ref[:n], dtype=np.float64)
    p_mod = np.asarray(ps_mod[:n], dtype=np.float64)
    mask  = np.isfinite(p_ref) & np.isfinite(p_mod) & (p_ref > 0) & (p_mod > 0)

    if not np.any(mask):
        return np.nan

    log_ratio = 10.0 * np.log10(
        np.maximum(p_ref[mask] / np.maximum(p_mod[mask], EPS), EPS)
    )
    return float(np.sqrt(np.mean(log_ratio ** 2)))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_rapsd(
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    obs_key: str = "Obs",
    title: str = "RAPSD",
    ax: plt.Axes | None = None,
    style_map: dict | None = None,
) -> plt.Axes:
    """
    Plot RAPSD curves for all models vs obs.

    Parameters
    ----------
    spectra   : {model_name: (k, ps)} dict — include obs under obs_key
    obs_key   : key in spectra that is the reference
    title     : plot title
    ax        : existing axes (created if None)
    style_map : {model_name: dict of line kwargs}
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    k_obs, ps_obs = spectra[obs_key]
    ax.semilogy(k_obs, ps_obs, color="black", lw=2.5, ls="-", label=obs_key)

    for name, (k, ps) in spectra.items():
        if name == obs_key:
            continue
        # compute RALSD for legend annotation
        score = ralsd(k_obs, ps_obs, k, ps)
        label = f"{name}  (RALSD={score:.2f} dB)"
        sty   = (style_map or {}).get(name, {})
        ax.semilogy(k, ps, label=label, **sty)

    ax.set_xlabel("Wavenumber (k)", fontsize=12)
    ax.set_ylabel("Power Spectral Density", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def spectral_ratio(
    da_pred: xr.DataArray,
    da_obs: xr.DataArray,
    time_dim: str | None = "time",
    sample_dim: str | None = "auto",
    normalize: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    k_pred, ps_pred = rapsd(da_pred, time_dim=time_dim, sample_dim=sample_dim,
                            normalize=normalize, desc="RAPSD pred")
    k_obs,  ps_obs  = rapsd(da_obs,  time_dim=time_dim, sample_dim=None,
                            normalize=normalize, desc="RAPSD obs")

    n     = min(k_pred.size, k_obs.size)
    if n == 0:
        raise ValueError("Empty spectra returned by rapsd().")

    # grids are identical (same domain) — direct slice, no interp needed
    ratio = ps_pred[:n] / np.maximum(ps_obs[:n], EPS)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.plot(k_pred[:n], ratio, lw=2, color="steelblue")
    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Spectral ratio (pred / obs)")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    return ax