from __future__ import annotations


import numpy as np
import xarray as xr


def _radial_psd_2d(field2d: np.ndarray) -> np.ndarray:
    f = np.asarray(field2d, dtype=np.float64)
    ny, nx = f.shape

    F = np.fft.fftshift(np.fft.fft2(f))
    P = (np.abs(F) ** 2) / (nx * ny)

    yy, xx = np.indices((ny, nx))
    cy, cx = ny // 2, nx // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)

    psd_r = np.full(rr.max() + 1, np.nan, dtype=np.float64)
    for k in range(psd_r.size):
        vals = P[rr == k]
        if vals.size:
            psd_r[k] = np.nanmean(vals)

    return psd_r[1:] 


def ralsd(pred: xr.DataArray, ref: xr.DataArray, eps=1e-12) -> float:
    pred, ref = xr.align(pred, ref, join="inner")
    if pred.sizes.get("time", 0) == 0:
        return np.nan

    daily = []
    for t in range(pred.sizes["time"]):
        pf = np.asarray(pred.isel(time=t).values, dtype=np.float64)
        rf = np.asarray(ref.isel(time=t).values, dtype=np.float64)

        m = np.isfinite(pf) & np.isfinite(rf)
        if m.sum() < 16:
            continue

        pf = np.where(m, pf, 0.0)
        rf = np.where(m, rf, 0.0)

        pf = np.where(m, pf - np.mean(pf[m]), 0.0)
        rf = np.where(m, rf - np.mean(rf[m]), 0.0)

        sp = _radial_psd_2d(pf)
        sr = _radial_psd_2d(rf)
        n = min(sp.size, sr.size)
        if n < 2:
            continue

        sp, sr = sp[:n], sr[:n]
        good = np.isfinite(sp) & np.isfinite(sr) & (sp >= 0) & (sr >= 0)
        if good.sum() < 2:
            continue

        day_val = np.mean((np.log(np.maximum(sr[good], eps)) - np.log(np.maximum(sp[good], eps))) ** 2)
        daily.append(float(day_val))

    return float(np.mean(daily)) if daily else np.nan

