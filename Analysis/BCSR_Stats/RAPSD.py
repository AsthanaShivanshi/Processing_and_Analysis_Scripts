from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

EPS = 1e-12
_NON_SPATIAL_HINTS = {"time", "member", "sample", "realization", "ensemble", "ens"}


def get_spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    """Infer 2D spatial dims."""
    preferred = [("rlat", "rlon"), ("lat", "lon"), ("y", "x"), ("latitude", "longitude")]
    for d0, d1 in preferred:
        if d0 in da.dims and d1 in da.dims:
            return d0, d1

    candidates = [d for d in da.dims if d not in _NON_SPATIAL_HINTS]
    if len(candidates) < 2:
        raise ValueError(f"Cannot infer spatial dims from {da.dims}")
    return candidates[-2], candidates[-1]


def _valid_mask2d(mask: xr.DataArray, sy: str, sx: str) -> np.ndarray:
    m = mask.squeeze(drop=True).transpose(sy, sx).values
    return np.isfinite(m) & (m != 0)


def _radial_setup(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    window = np.outer(np.hanning(ny), np.hanning(nx)).astype(np.float64)

    yy, xx = np.indices((ny, nx))
    cy, cx = ny // 2, nx // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32).ravel()

    nbins = int(rr.max()) + 1
    counts = np.bincount(rr, minlength=nbins).astype(np.float64)
    counts[counts == 0] = 1.0

    k = np.arange(nbins, dtype=np.float64) / float(max(ny, nx))
    return window, rr, counts, k


def _iter_realization_blocks(da: xr.DataArray, sy: str, sx: str, batch_size: int):
    """Yield blocks shaped (n, ny, nx), flattening all non-spatial dims."""
    non_spatial = [d for d in da.dims if d not in (sy, sx)]
    if not non_spatial:
        arr = da.transpose(sy, sx).values.astype(np.float64)
        yield arr[None, :, :]
        return

    da_t = da.transpose(*non_spatial, sy, sx)
    lead = "time" if "time" in non_spatial else non_spatial[0]
    n = da_t.sizes[lead]

    for i in range(0, n, batch_size):
        slab = da_t.isel({lead: slice(i, min(i + batch_size, n))}).values.astype(np.float64)
        ny, nx = slab.shape[-2], slab.shape[-1]
        yield slab.reshape(-1, ny, nx)


def _radial_profiles_block(
    block: np.ndarray,
    valid2d: np.ndarray,
    window: np.ndarray,
    rindex: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """RAPSD profiles for one block, output shape (n, nk)."""
    z = np.asarray(block, dtype=np.float64)
    if z.ndim != 3 or z.shape[0] == 0:
        return np.empty((0, counts.size), dtype=np.float64)

    inside = valid2d[None, :, :]
    finite_inside = np.isfinite(z) & inside
    n_valid = finite_inside.sum(axis=(1, 2))
    keep = n_valid > 0
    if not np.any(keep):
        return np.empty((0, counts.size), dtype=np.float64)

    z = z[keep].copy()
    finite_inside = finite_inside[keep]
    n_valid = n_valid[keep].astype(np.float64)

    means = np.where(finite_inside, z, 0.0).sum(axis=(1, 2)) / n_valid

    bad_inside = (~np.isfinite(z)) & inside
    if np.any(bad_inside):
        ii = np.where(bad_inside)[0]
        z[bad_inside] = means[ii]

    z[:, ~valid2d] = 0.0
    z[:, valid2d] -= means[:, None]
    z *= window[None, :, :]

    fft2 = np.fft.fftshift(np.fft.fft2(z, axes=(-2, -1)), axes=(-2, -1))
    power = np.abs(fft2) ** 2
    power_flat = power.reshape(power.shape[0], -1)

    out = np.empty((power.shape[0], counts.size), dtype=np.float64)
    for i in range(power.shape[0]):
        out[i] = np.bincount(rindex, weights=power_flat[i], minlength=counts.size) / counts
    return out


def stream_mean_isotropic_spectrum(
    da: xr.DataArray,
    hr_mask: xr.DataArray,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Mean RAPSD across all realizations pooled from non-spatial dims.
    Returns (k, mean_power, n_realizations_used).
    """
    sy, sx = get_spatial_dims(da)
    valid2d = _valid_mask2d(hr_mask, sy, sx)
    if not np.any(valid2d):
        return np.array([]), np.array([]), 0

    ny, nx = valid2d.shape
    window, rindex, counts, k = _radial_setup(ny, nx)

    ps_sum = np.zeros_like(k)
    n_used = 0

    for block in _iter_realization_blocks(da, sy, sx, batch_size=batch_size):
        if block.shape[-2:] != (ny, nx):
            raise ValueError(f"Field shape {block.shape[-2:]} does not match mask {(ny, nx)}")
        radial = _radial_profiles_block(block, valid2d, window, rindex, counts)
        if radial.size == 0:
            continue
        ps_sum += np.nansum(radial, axis=0)
        n_used += radial.shape[0]

    if n_used == 0:
        return np.array([]), np.array([]), 0

    ps_mean = ps_sum / float(n_used)
    keep = (k > 0) & np.isfinite(ps_mean) & (ps_mean > 0)
    return k[keep], ps_mean[keep], n_used


def ralsd(k_ref: np.ndarray, ps_ref: np.ndarray, k_mod: np.ndarray, ps_mod: np.ndarray) -> float:
    """Radially Averaged Log Spectral Distance (lower is better)."""
    if k_ref.size == 0 or k_mod.size == 0:
        return np.nan

    kmin = max(np.nanmin(k_ref), np.nanmin(k_mod))
    kmax = min(np.nanmax(k_ref), np.nanmax(k_mod))
    if not np.isfinite(kmin) or not np.isfinite(kmax) or kmax <= kmin:
        return np.nan

    k_common = k_ref[(k_ref >= kmin) & (k_ref <= kmax)]
    if k_common.size == 0:
        return np.nan

    p_ref = np.interp(k_common, k_ref, ps_ref)
    p_mod = np.interp(k_common, k_mod, ps_mod)

    m = np.isfinite(p_ref) & np.isfinite(p_mod) & (p_ref > 0) & (p_mod > 0)
    if not np.any(m):
        return np.nan

    d = np.log10(p_mod[m] + EPS) - np.log10(p_ref[m] + EPS)
    return float(np.sqrt(np.mean(d**2)))


def plot_mean_rapsd_with_ralsd(
    pred: xr.DataArray,
    ref: xr.DataArray,
    hr_mask: xr.DataArray,
    *,
    batch_size: int = 8,
    title: str = "RAPSD comparison",
    save_path: str | Path | None = None,
    show: bool = True,
) -> float:
    """Single model vs obs: mean RAPSD curves + RALSD in title."""
    k_ref, ps_ref, n_ref = stream_mean_isotropic_spectrum(ref, hr_mask, batch_size=batch_size)
    k_mod, ps_mod, n_mod = stream_mean_isotropic_spectrum(pred, hr_mask, batch_size=batch_size)
    score = ralsd(k_ref, ps_ref, k_mod, ps_mod)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    if k_ref.size:
        ax.loglog(k_ref, ps_ref, color="black", lw=2.0, label=f"Obs (n={n_ref})")
    if k_mod.size:
        ax.loglog(k_mod, ps_mod, color="tab:blue", lw=2.0, label=f"Model (n={n_mod})")

    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Power")
    ax.set_title(f"{title} | RALSD={score:.4f}" if np.isfinite(score) else f"{title} | RALSD=NaN")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return score


def plot_all_baselines_rapsd(
    k_ref: np.ndarray,
    ps_ref: np.ndarray,
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    scores: dict[str, float] | None = None,
    *,
    title: str = "RAPSD: all baselines",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Obs + all baseline mean RAPSD curves in one figure."""
    scores = scores or {}
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)

    if k_ref.size:
        ax.loglog(k_ref, ps_ref, color="black", lw=2.4, label="MCH (obs)")

    for name, (k, ps) in spectra.items():
        s = scores.get(name, np.nan)
        label = f"{name} | RALSD={s:.3f}" if np.isfinite(s) else name
        ax.loglog(k, ps, lw=1.2, alpha=0.95, label=label)

    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Power")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=False)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)