from __future__ import annotations

import gc
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm

EPS = 1e-30
SAMPLE_DIM_CANDIDATES = ("member", "sample", "samples")

# Harris et al. (2022): drop frames with domain-mean rain < 0.002 mm/hr.
# Daily data -> 0.002 * 24 = 0.048 mm/day. Import this in rapsd_method_*.py
# too, so the threshold cannot drift between scripts.
PRECIP_MIN_MEAN = 0.05


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
    """
    Rounded radial index per pixel, matching rapsd_method_I._compute_rapsd.

    NOTE: method_I uses np.round (nearest bin), NOT floor. The DC component
    sits at (h//2, w//2) after fftshift.
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.round(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)).astype(np.int32)
    r = np.ascontiguousarray(r)
    r.flags.writeable = False
    return r


@lru_cache(maxsize=16)
def _max_radius(shape: tuple[int, int]) -> int:
    """Inscribed-circle cutoff (method_I): bins beyond this are corner-only."""
    return min(shape) // 2


@lru_cache(maxsize=16)
def _bin_counts(shape: tuple[int, int]) -> np.ndarray:
    nr = np.bincount(_radial_bins(shape).ravel())
    nr.flags.writeable = False
    return nr


def _radial_average(array_2d: np.ndarray) -> np.ndarray:
    """
    Mean of `array_2d` over each rounded-radius annulus, truncated to the
    inscribed-circle radius. Equivalent to method_I's per-annulus loop,
    but vectorised via bincount (~50x faster).
    """
    shape = array_2d.shape
    r = _radial_bins(shape)
    nr = _bin_counts(shape)

    prof = np.bincount(r.ravel(), weights=array_2d.ravel(), minlength=nr.size)
    prof = prof / np.maximum(nr, 1)
    return prof[: _max_radius(shape)]


def _field_profile(field_2d: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Per-frame RAPSD, equivalent to rapsd_method_I._compute_rapsd()[0].

    Returns an empty array for frames that should be skipped
    (too few valid pixels, or near-constant e.g. dry precip days).
    """
    img = np.asarray(field_2d, dtype=np.float64)

    valid = np.isfinite(img)
    if valid.sum() < 4:
        return np.array([])

    # Fill masked/NaN pixels with the mean of the VALID pixels (not 0.0):
    # zero-filling imprints the domain outline into the spectrum.
    img = np.where(valid, img, img[valid].mean())

    # Harris et al. 2022: subtract mean to suppress the DC component
    img = img - img.mean()

    # skip near-constant frames (e.g. dry precip days)
    if np.std(img) < 1e-6:
        return np.array([])

    h, w = img.shape
    power = (np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2) / (h * w)

    profile = _radial_average(power)

    if normalize:
        s = profile.sum()
        if s > 0:
            profile = profile / s

    return profile


def wavenumber_to_wavelength_km(
    k: np.ndarray, shape: tuple[int, int], grid_spacing_km: float = 1.0
) -> np.ndarray:
    """
    Convert the integer radial wavenumber returned by rapsd() into a
    wavelength in km, using method_I's convention:

        freq = k / max_radius * 0.5      [cycles / pixel]
        wavelength = grid_spacing / freq [km]
    """
    max_r = _max_radius(tuple(shape))
    freq = np.asarray(k, dtype=np.float64) / max_r * 0.5
    with np.errstate(divide="ignore"):
        return grid_spacing_km / freq


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def rapsd(
    da: xr.DataArray,
    time_dim: str | None = "time",
    sample_dim: str | None = "auto",
    normalize: bool = False,
    desc: str = "RAPSD",
    chunk_size: int = 50,            # frames per chunk — controls OOM
    min_frame_mean: float | None = None,
    return_counts: bool = False,
):
    """
    Compute the mean Radially Averaged Power Spectral Density.

    The per-frame kernel matches rapsd_method_I._compute_rapsd exactly
    (mean-fill of NaNs, de-meaning, |FFT|^2/(h*w), rounded radial bins,
    inscribed-circle truncation).

    Iterates over frames one chunk at a time to avoid loading the full
    array into memory (OOM-safe). This function is inherently *unpaired*:
    each dataset's spectrum is computed independently, so obs and model
    need not share a time axis.

    Parameters
    ----------
    da             : input DataArray  (time [x sample] x lat x lon)
    time_dim       : name of the time dimension
    sample_dim     : name of the ensemble/sample dimension or 'auto'
    normalize      : normalize each spectrum by its total power before
                     averaging. Default False to match method_I, which
                     averages ABSOLUTE power.
    desc           : tqdm label
    chunk_size     : number of time steps loaded into RAM at once
    min_frame_mean : if set, skip frames whose valid-pixel mean is below this
                     value (use PRECIP_MIN_MEAN for precipitation). Applied
                     per-dataset, preserving unpaired evaluation.
    return_counts  : also return a dict of frame counts, needed to report the
                     unpaired sample sizes in figure captions.

    Returns
    -------
    k       : wavenumber array (positive integers, k > 0)
    ps_mean : mean radial power spectrum
    counts  : (only if return_counts) dict of frame counts
    """
    sd = _resolve_sample_dim(da, sample_dim)

    # identify spatial dims
    real_dims    = [d for d in [time_dim, sd] if d is not None and d in da.dims]
    spatial_dims = [d for d in da.dims if d not in real_dims]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected 2 spatial dims, got {spatial_dims}")

    # reorder so we iterate over leading dims
    da = da.transpose(*real_dims, *spatial_dims)

    has_time = bool(time_dim) and time_dim in da.dims
    n_time   = da.sizes[time_dim] if has_time else 1
    n_sample = da.sizes[sd] if sd and sd in da.dims else 1
    n_frames = n_time * n_sample

    profiles: list[np.ndarray] = []
    skipped = 0
    dry_skipped = 0

    with tqdm(total=n_frames, desc=desc, unit="frame", leave=False) as pbar:
        for t_start in range(0, n_time, chunk_size):
            t_end = min(t_start + chunk_size, n_time)
            # load only this chunk — avoids full-array materialisation.
            # float64 throughout, matching method_I (no float32 round-trip).
            sub = da.isel({time_dim: slice(t_start, t_end)}) if has_time else da
            chunk = np.asarray(sub.values, dtype=np.float64)
            # reshape to (frames, H, W)
            chunk = chunk.reshape(-1, chunk.shape[-2], chunk.shape[-1])

            for frame in chunk:
                if min_frame_mean is not None:
                    finite = np.isfinite(frame)
                    if finite.sum() == 0 or frame[finite].mean() < min_frame_mean:
                        dry_skipped += 1
                        pbar.update(1)
                        continue

                profile = _field_profile(frame, normalize)
                if profile.size == 0:
                    skipped += 1
                else:
                    profiles.append(profile)
                pbar.update(1)

            # free chunk memory explicitly
            del chunk
            gc.collect()

    if dry_skipped:
        print(f"[rapsd] {dry_skipped}/{n_frames} frames below min_frame_mean")
    if skipped:
        print(f"[rapsd] skipped {skipped}/{n_frames} near-constant/empty frames")

    if not profiles:
        raise RuntimeError("All frames were skipped — no valid spectra computed.")

    min_len = min(p.size for p in profiles)
    stack   = np.stack([p[:min_len] for p in profiles], axis=0)
    ps_mean = stack.mean(axis=0)
    k       = np.arange(min_len, dtype=np.float64)

    keep = (k > 0) & np.isfinite(ps_mean) & (ps_mean > 0)

    if return_counts:
        counts = {
            "n_frames": n_frames,
            "n_used": len(profiles),
            "n_below_threshold": dry_skipped,
            "n_near_constant": skipped,
        }
        return k[keep], ps_mean[keep], counts
    return k[keep], ps_mean[keep]


def ralsd(
    k_ref: np.ndarray,
    ps_ref: np.ndarray,
    k_mod: np.ndarray,
    ps_mod: np.ndarray,
) -> float:
    """
    RALSD = sqrt( mean( (10 log10( S_true / S_pred ))^2 ) )
    Harris et al. (2022).

    Both spectra must come from rapsd() calls with the SAME `normalize` flag.
    The log-ratio is squared, so the ref/mod ordering is irrelevant —
    this matches rapsd_method_I._ralsd numerically.
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
    field_shape: tuple[int, int] | None = None,
    grid_spacing_km: float = 1.0,
) -> plt.Axes:
    """
    Plot RAPSD curves for all models vs obs.

    Parameters
    ----------
    spectra     : {model_name: (k, ps)} dict — include obs under obs_key
    obs_key     : key in spectra that is the reference
    title       : plot title
    ax          : existing axes (created if None)
    style_map   : {model_name: dict of line kwargs}
    field_shape : if given, the x-axis is converted to wavelength in km
                  (comparable with the rapsd_method_I figures)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    def _x(k):
        if field_shape is None:
            return k
        return wavenumber_to_wavelength_km(k, tuple(field_shape), grid_spacing_km)

    k_obs, ps_obs = spectra[obs_key]
    ax.semilogy(_x(k_obs), ps_obs, color="black", lw=2.5, ls="-", label=obs_key)

    for name, (k, ps) in spectra.items():
        if name == obs_key:
            continue
        # compute RALSD for legend annotation
        score = ralsd(k_obs, ps_obs, k, ps)
        label = f"{name}  (RALSD={score:.2f} dB)"
        sty   = (style_map or {}).get(name, {})
        ax.semilogy(_x(k), ps, label=label, **sty)

    if field_shape is None:
        ax.set_xlabel("Wavenumber (k)", fontsize=12)
    else:
        ax.set_xlabel("Wavelength (km)", fontsize=12)
        ax.invert_xaxis()

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
    normalize: bool = False,
    min_frame_mean: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    k_pred, ps_pred = rapsd(da_pred, time_dim=time_dim, sample_dim=sample_dim,
                            normalize=normalize, min_frame_mean=min_frame_mean,
                            desc="RAPSD pred")
    k_obs,  ps_obs  = rapsd(da_obs,  time_dim=time_dim, sample_dim=None,
                            normalize=normalize, min_frame_mean=min_frame_mean,
                            desc="RAPSD obs")

    n = min(k_pred.size, k_obs.size)
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