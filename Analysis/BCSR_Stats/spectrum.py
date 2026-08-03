from __future__ import annotations

import numpy as np
import xarray as xr

EPS = 1e-12


def get_spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    preferred = [("rlat", "rlon"), ("lat", "lon"), ("y", "x"), ("latitude", "longitude")]
    for d0, d1 in preferred:
        if d0 in da.dims and d1 in da.dims:
            return d0, d1
    non_sample = [d for d in da.dims if d not in ("time", "member", "sample", "realization", "ensemble", "ens")]
    if len(non_sample) < 2:
        raise ValueError(f"Cannot infer 2D spatial dims from {da.dims}")
    return non_sample[-2], non_sample[-1]


def _valid_mask2d(mask: xr.DataArray, sy: str, sx: str) -> np.ndarray:
    m = mask.squeeze(drop=True).transpose(sy, sx)
    mv = np.asarray(m.values)
    return np.isfinite(mv) & (mv != 0)


def _radial_setup(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    win = np.outer(np.hanning(ny), np.hanning(nx)).astype(np.float64)
    yy, xx = np.indices((ny, nx))
    cy, cx = ny // 2, nx // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    rflat = rr.ravel()
    nbins = int(rflat.max()) + 1
    counts = np.bincount(rflat, minlength=nbins).astype(np.float64)
    counts[counts == 0] = 1.0
    k = np.arange(nbins, dtype=np.float64) / float(max(ny, nx))
    return win, rflat, counts, k


def _iter_sample_blocks(da: xr.DataArray, sy: str, sx: str, batch_size: int):
    sample_dims = [d for d in da.dims if d not in (sy, sx)]
    if not sample_dims:
        arr = np.asarray(da.transpose(sy, sx).values, dtype=np.float64)
        yield arr[None, :, :]
        return

    da_t = da.transpose(*sample_dims, sy, sx)
    lead_dim = "time" if "time" in sample_dims else sample_dims[0]
    n_lead = da_t.sizes[lead_dim]

    for start in range(0, n_lead, batch_size):
        slab = da_t.isel({lead_dim: slice(start, min(start + batch_size, n_lead))}).values
        arr = np.asarray(slab, dtype=np.float64)
        ny, nx = arr.shape[-2], arr.shape[-1]
        yield arr.reshape(-1, ny, nx)


def _block_radial_profiles(
    block: np.ndarray,
    valid2d: np.ndarray,
    win: np.ndarray,
    rflat: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
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

    sums_inside = np.where(finite_inside, z, 0.0).sum(axis=(1, 2))
    means_inside = sums_inside / n_valid

    bad_inside = (~np.isfinite(z)) & inside
    if np.any(bad_inside):
        samp_idx = np.where(bad_inside)[0]
        z[bad_inside] = means_inside[samp_idx]

    z[:, ~valid2d] = 0.0
    z[:, valid2d] = z[:, valid2d] - means_inside[:, None]
    z *= win[None, :, :]

    fft2 = np.fft.fftshift(np.fft.fft2(z, axes=(-2, -1)), axes=(-2, -1))
    power = np.abs(fft2) ** 2
    power_flat = power.reshape(power.shape[0], -1)

    radial = np.empty((power.shape[0], counts.size), dtype=np.float64)
    for i in range(power.shape[0]):
        radial[i] = np.bincount(rflat, weights=power_flat[i], minlength=counts.size) / counts
    return radial


def stream_mean_isotropic_spectrum(
    da: xr.DataArray,
    hr_mask: xr.DataArray,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, int]:
    sy, sx = get_spatial_dims(da)
    valid2d = _valid_mask2d(hr_mask, sy, sx)
    if not np.any(valid2d):
        return np.array([]), np.array([]), 0

    ny, nx = valid2d.shape
    win, rflat, counts, k = _radial_setup(ny, nx)

    ps_sum = np.zeros_like(k)
    n_used = 0

    for block in _iter_sample_blocks(da, sy, sx, batch_size=batch_size):
        if block.shape[-2:] != (ny, nx):
            raise ValueError(f"Field shape {block.shape[-2:]} does not match mask shape {(ny, nx)}")
        radial = _block_radial_profiles(block, valid2d, win, rflat, counts)
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
    if k_ref.size == 0 or k_mod.size == 0 or ps_ref.size == 0 or ps_mod.size == 0:
        return np.nan

    if k_ref.shape == k_mod.shape and np.allclose(k_ref, k_mod):
        kr, pr, pm = k_ref, ps_ref, ps_mod
    else:
        kmin = max(float(np.nanmin(k_ref)), float(np.nanmin(k_mod)))
        kmax = min(float(np.nanmax(k_ref)), float(np.nanmax(k_mod)))
        kr = k_ref[(k_ref >= kmin) & (k_ref <= kmax)]
        if kr.size == 0:
            return np.nan
        pr = np.interp(kr, k_ref, ps_ref)
        pm = np.interp(kr, k_mod, ps_mod)

    mask = np.isfinite(kr) & np.isfinite(pr) & np.isfinite(pm) & (kr > 0) & (pr > 0) & (pm > 0)
    if not np.any(mask):
        return np.nan

    d = np.log10(pm[mask] + EPS) - np.log10(pr[mask] + EPS)
    return float(np.sqrt(np.mean(d ** 2)))