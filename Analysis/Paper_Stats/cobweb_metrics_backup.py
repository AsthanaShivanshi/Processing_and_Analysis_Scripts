from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from numpy.fft import fft2, fftshift


from properscoring import crps_ensemble
from scipy.ndimage import uniform_filter


from skimage.metrics import structural_similarity
from tqdm import tqdm

sns.set_style("whitegrid")

SCRIPT_PATH = Path(__file__).resolve()
PAPER_STATS_DIR = SCRIPT_PATH.parent
PROCESSING_ROOT = PAPER_STATS_DIR.parent.parent.parent

START = "2015-01-01"
END = "2023-12-31"
SAMPLE_DIMS = ("member", "sample", "samples")
EPS = 1e-30

GRID_SPACING_KM = 1.0
NYQUIST_KM = 2.0 * GRID_SPACING_KM
PRECIP_MIN_MEAN = 0.05  # mm/day (Harris et al. 0.002 mm/hr * 24)

# FSS settings for paired precipitation fields

FSS_THRESHOLD_MM_DAY = 1.0

FSS_WINDOW_KM = 4.0
FSS_WINDOW_PX = max(1, int(round(FSS_WINDOW_KM / GRID_SPACING_KM)))


if FSS_WINDOW_PX % 2 == 0:
    FSS_WINDOW_PX += 1


def _load_field(path: Path, var: str, mask: xr.DataArray | None = None, clip: bool = False) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        da = ds[var].sel(time=slice(START, END)).load()
    if clip:
        da = da.clip(min=0)
    return da.where(mask) if mask is not None else da


def _load_mask(path: Path, var: str) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        return ds[var].load()


def _sample_dim(da: xr.DataArray) -> str | None:
    return next((d for d in SAMPLE_DIMS if d in da.dims), None)


def _spatial_mask(mask2d: xr.DataArray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask2d is None:
        return np.ones(shape, dtype=bool)
    mv = mask2d.values
    return np.isfinite(mv) & (mv != 0)


def _finite(x) -> np.ndarray:
    a = np.asarray(list(x), dtype=float)
    return a[np.isfinite(a)]


def _mean(x) -> float:
    a = np.asarray(list(x), dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else np.nan


def _frame_to_ensemble(pred_frame: xr.DataArray) -> np.ndarray:
    sd = _sample_dim(pred_frame)
    if sd is None:
        return pred_frame.values[None, ...]
    spatial = [d for d in pred_frame.dims if d != sd]
    return pred_frame.transpose(sd, *spatial).values


# ── RAPSD ─────────────────────────────────────────────────────────────────────
def _compute_rapsd_raw(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img = img.astype(np.float64)
    valid = np.isfinite(img)
    if valid.sum() < 4:
        return np.array([]), np.array([])

    field_mean = img[valid].mean()
    img = np.where(valid, img, field_mean)
    img = img - img.mean()

    if np.std(img) < 1e-6:
        return np.array([]), np.array([])

    h, w = img.shape
    power = (np.abs(fftshift(fft2(img))) ** 2) / (h * w)

    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_radius = min(h, w) // 2

    radius_int = np.round(radius).astype(int)
    rapsd = np.zeros(max_radius)
    for r in range(max_radius):
        annulus = radius_int == r
        if annulus.sum() > 0:
            rapsd[r] = np.mean(power[annulus])

    frequencies = np.arange(max_radius, dtype=float) / max_radius * 0.5
    return rapsd, frequencies


def _get_valid_precip_times(obs: xr.DataArray) -> np.ndarray:
    nt = obs.sizes.get("time", 1)
    valid = []
    for t in range(nt):
        frame = obs.isel(time=t).values.astype(float)
        finite = np.isfinite(frame)
        if finite.sum() > 0 and np.nanmean(frame[finite]) >= PRECIP_MIN_MEAN:
            valid.append(t)
    return np.array(valid, dtype=int)


def _rapsd_timemean(
    da: xr.DataArray,
    valid_times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sd = _sample_dim(da)
    nt = da.sizes.get("time", 1)
    time_indices = valid_times if valid_times is not None else np.arange(nt)

    spectra: list[np.ndarray] = []
    freqs: np.ndarray | None = None

    for t in time_indices:
        frame_da = da.isel(time=int(t)) if "time" in da.dims else da
        if sd is not None and sd in frame_da.dims:
            for s in range(frame_da.sizes[sd]):
                frame = frame_da.isel({sd: s}).values.astype(float)
                r, f = _compute_rapsd_raw(frame)
                if r.size > 0:
                    spectra.append(r)
                    freqs = f
        else:
            frame = frame_da.values.astype(float)
            r, f = _compute_rapsd_raw(frame)
            if r.size > 0:
                spectra.append(r)
                freqs = f

    if not spectra or freqs is None:
        return np.array([]), np.array([])

    return np.mean(spectra, axis=0), freqs


def _ralsd_from_timemean(
    obs: xr.DataArray,
    pred: xr.DataArray,
    is_precip: bool = False,
) -> float:
    valid_times = _get_valid_precip_times(obs) if is_precip else None

    rapsd_obs, freqs = _rapsd_timemean(obs, valid_times=valid_times)
    rapsd_pred, _ = _rapsd_timemean(pred, valid_times=valid_times)

    if rapsd_obs.size == 0 or rapsd_pred.size == 0 or freqs.size == 0:
        return np.nan

    valid = (freqs > 0) & (1.0 / (freqs / GRID_SPACING_KM) >= NYQUIST_KM)
    ps_obs = rapsd_obs[valid]
    ps_pred = rapsd_pred[valid]

    n = min(ps_obs.size, ps_pred.size)
    ps_obs = ps_obs[:n]
    ps_pred = ps_pred[:n]

    if n == 0:
        return np.nan

    d = 10.0 * np.log10(np.maximum(ps_pred, EPS) / np.maximum(ps_obs, EPS))
    return float(np.sqrt(np.sum(d ** 2) / n))


# ── FSS ───────────────────────────────────────────────────────────────────────
def _fss_2d(
    obs2d: np.ndarray,
    pred2d: np.ndarray,
    mask2d: xr.DataArray | None = None,
    threshold: float = FSS_THRESHOLD_MM_DAY,
    window_px: int = FSS_WINDOW_PX,
) -> float:
    valid = np.isfinite(obs2d) & np.isfinite(pred2d)
    if mask2d is not None:
        valid &= _spatial_mask(mask2d, obs2d.shape)

    if not np.any(valid):
        return np.nan

    valid_f = valid.astype(float)
    normalizer = uniform_filter(valid_f, size=window_px, mode="constant", cval=0.0)
    active = normalizer > 0
    if not np.any(active):
        return np.nan

    obs_bin = ((obs2d >= threshold) & valid).astype(float)
    pred_bin = ((pred2d >= threshold) & valid).astype(float)

    obs_sum = uniform_filter(obs_bin, size=window_px, mode="constant", cval=0.0)
    pred_sum = uniform_filter(pred_bin, size=window_px, mode="constant", cval=0.0)

    obs_frac = np.where(active, obs_sum / np.maximum(normalizer, EPS), np.nan)
    pred_frac = np.where(active, pred_sum / np.maximum(normalizer, EPS), np.nan)

    ok = np.isfinite(obs_frac) & np.isfinite(pred_frac)
    if not np.any(ok):
        return np.nan

    mse = np.mean((pred_frac[ok] - obs_frac[ok]) ** 2)
    denom = np.mean(pred_frac[ok] ** 2 + obs_frac[ok] ** 2)
    return float(1.0 - mse / denom) if denom > 0 else np.nan


def _series_fss(obs: xr.DataArray, pred: xr.DataArray, mask2d=None) -> float:
    sd = _sample_dim(pred)
    if sd is not None:
        pred = pred.mean(dim=sd, skipna=True)
    obs, pred = xr.align(obs, pred, join="inner")
    tmax = min(obs.sizes.get("time", 0), pred.sizes.get("time", 0))
    return _mean(
        _fss_2d(obs.isel(time=t).values, pred.isel(time=t).values, mask2d)
        for t in range(tmax)
    )


# ── Other metrics ────────────────────────────────────────────────────────────
def _rmse_2d(obs2d: np.ndarray, pred2d: np.ndarray, mask2d: xr.DataArray | None = None) -> float:
    valid = np.isfinite(obs2d) & np.isfinite(pred2d)
    if mask2d is not None:
        valid &= _spatial_mask(mask2d, obs2d.shape)
    return float(np.sqrt(np.mean((pred2d[valid] - obs2d[valid]) ** 2))) if np.any(valid) else np.nan


def _mae_2d(obs2d: np.ndarray, pred2d: np.ndarray, mask2d: xr.DataArray | None = None) -> float:
    valid = np.isfinite(obs2d) & np.isfinite(pred2d)
    if mask2d is not None:
        valid &= _spatial_mask(mask2d, obs2d.shape)
    return float(np.mean(np.abs(pred2d[valid] - obs2d[valid]))) if np.any(valid) else np.nan


def _ssim_2d(obs2d: np.ndarray, pred2d: np.ndarray, mask2d: xr.DataArray | None = None) -> float:
    valid = np.isfinite(obs2d) & np.isfinite(pred2d)
    if mask2d is not None:
        valid &= _spatial_mask(mask2d, obs2d.shape)
    if not np.any(valid):
        return np.nan

    o_fill = np.nanmean(obs2d[valid])
    p_fill = np.nanmean(pred2d[valid])
    o = np.where(valid, obs2d, o_fill)
    p = np.where(valid, pred2d, p_fill)
    dr = o[valid].max() - o[valid].min()

    if dr == 0:
        return 1.0 if np.allclose(o, p, equal_nan=True) else 0.0

    try:
        return float(structural_similarity(o, p, data_range=dr))
    except Exception:
        return np.nan


def _crps_frame(obs2d: np.ndarray, pred_frame: xr.DataArray, mask2d: xr.DataArray | None = None) -> float:
    ens3d = _frame_to_ensemble(pred_frame)
    valid = np.isfinite(obs2d) & np.all(np.isfinite(ens3d), axis=0)
    if mask2d is not None:
        valid &= _spatial_mask(mask2d, obs2d.shape)
    if not np.any(valid):
        return np.nan
    return float(np.mean(crps_ensemble(obs2d[valid], ens3d[:, valid].T)))


def _pitd_frame(
    obs2d: np.ndarray,
    pred_frame: xr.DataArray,
    mask2d: xr.DataArray | None = None,
    bins: int = 50,
    seed: int = 0,
) -> float:
    ens3d = _frame_to_ensemble(pred_frame)
    valid = np.isfinite(obs2d) & np.all(np.isfinite(ens3d), axis=0)
    if mask2d is not None:
        valid &= _spatial_mask(mask2d, obs2d.shape)
    if not np.any(valid):
        return np.nan

    obs = obs2d[valid]
    ens = ens3d[:, valid]
    less = np.mean(ens < obs[None, :], axis=0)
    equal = np.mean(np.isclose(ens, obs[None, :], rtol=0.0, atol=1e-6), axis=0)

    pit = less + np.random.default_rng(seed).random(obs.size) * equal
    hist, _ = np.histogram(pit, bins=np.linspace(0, 1, bins + 1))
    if hist.sum() == 0:
        return np.nan

    p = hist / hist.sum()
    u = np.ones(bins) / bins
    return float(np.sqrt(np.mean((p - u) ** 2)))


def _series_mean(obs: xr.DataArray, pred: xr.DataArray, fn, mask2d=None) -> float:
    sd = _sample_dim(pred)
    if sd is not None:
        pred = pred.mean(dim=sd, skipna=True)
    obs, pred = xr.align(obs, pred, join="inner")
    tmax = min(obs.sizes.get("time", 0), pred.sizes.get("time", 0))
    return _mean(fn(obs.isel(time=t).values, pred.isel(time=t).values, mask2d) for t in range(tmax))


def _series_crps(obs: xr.DataArray, pred: xr.DataArray, mask2d=None) -> float:
    obs, pred = xr.align(obs, pred, join="inner")
    tmax = min(obs.sizes.get("time", 0), pred.sizes.get("time", 0))
    return _mean(_crps_frame(obs.isel(time=t).values, pred.isel(time=t), mask2d) for t in range(tmax))


def _series_pitd(obs: xr.DataArray, pred: xr.DataArray, mask2d=None, bins: int = 20) -> float:
    obs, pred = xr.align(obs, pred, join="inner")
    tmax = min(obs.sizes.get("time", 0), pred.sizes.get("time", 0))
    return _mean(_pitd_frame(obs.isel(time=t).values, pred.isel(time=t), mask2d, bins=bins, seed=t) for t in range(tmax))


def _row(
    obs: xr.DataArray,
    pred: xr.DataArray,
    mask2d: xr.DataArray,
    model: str,
    variable: str,
) -> dict[str, float | str]:
    is_precip = variable == "precip"

    ralsd_mean = _ralsd_from_timemean(obs, pred, is_precip=is_precip)

    out: dict[str, float | str] = {
        "model": model,
        "variable": variable,
        "CRPS": _series_crps(obs, pred, mask2d),
        "RALSD": ralsd_mean,
        "SSIM": _series_mean(obs, pred, _ssim_2d, mask2d),
        "RMSE": _series_mean(obs, pred, _rmse_2d, mask2d),
        "MAE": _series_mean(obs, pred, _mae_2d, mask2d),
        "PITD": _series_pitd(obs, pred, mask2d),
        "FSS": _series_fss(obs, pred, mask2d) if is_precip else np.nan,
    }
    return out


def main() -> None:
    mask_lr = _load_mask(
        PROCESSING_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_LR.nc",
        "TabsD",
    )
    mask_hr = _load_mask(
        PROCESSING_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc",
        "TabsD",
    )

    obs_temp = _load_field(
        PROCESSING_ROOT / "Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/TabsD_1971_2023.nc",
        "TabsD",
    )
    obs_precip = _load_field(
        PROCESSING_ROOT / "Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/RhiresD_1971_2023.nc",
        "RhiresD",
        clip=True,
    )

    coarse_temp = _load_field(
        PROCESSING_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step2_coarse.nc",
        "TabsD",
        mask_lr,
    ).interp_like(mask_hr, method="nearest")
    coarse_precip = _load_field(
        PROCESSING_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step2_coarse.nc",
        "RhiresD",
        mask_lr,
        clip=True,
    ).interp_like(mask_hr, method="nearest")

    temp_models = {
        "Coarse": coarse_temp,
        "Bicubic": _load_field(
            PROCESSING_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step3_interp_bicubic.nc",
            "TabsD",
            mask_hr,
        ),
        "Bilinear": _load_field(
            PROCESSING_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step3_interp_bilinear.nc",
            "TabsD",
            mask_hr,
        ),
        "UNet": _load_field(
            PROCESSING_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/unet_downscaled_test_set_2015_2023.nc",
            "temp",
            mask_hr,
        ),
        "DDIM": _load_field(
            PROCESSING_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc",
            "temp",
            mask_hr,
        ),
        "CFM": _load_field(
            PROCESSING_ROOT / "Downscaling_Models/FM_conditional_derived/output_inference/fm_downscaled_test_set_allframes_steps10_samples10.nc",
            "temp",
            mask_hr,
        ),
    }

    precip_models = {
        "Coarse": coarse_precip,
        "Bicubic": _load_field(
            PROCESSING_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp_bicubic.nc",
            "RhiresD",
            mask_hr,
            clip=True,
        ),
        "Bilinear": _load_field(
            PROCESSING_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp_bilinear.nc",
            "RhiresD",
            mask_hr,
            clip=True,
        ),
        "UNet": _load_field(
            PROCESSING_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/unet_downscaled_test_set_2015_2023.nc",
            "precip",
            mask_hr,
            clip=True,
        ),
        "DDIM": _load_field(
            PROCESSING_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc",
            "precip",
            mask_hr,
            clip=True,
        ),
        "CFM": _load_field(
            PROCESSING_ROOT / "Downscaling_Models/FM_conditional_derived/output_inference/fm_downscaled_test_set_allframes_steps10_samples10.nc",
            "precip",
            mask_hr,
            clip=True,
        ),
    }

    rows = []
    for variable, obs, models in tqdm(
        [
            ("temp", obs_temp, temp_models),
            ("precip", obs_precip, precip_models),
        ],
        desc="variables",
    ):
        for name, pred in tqdm(models.items(), desc=variable, leave=False):
            rows.append(_row(obs, pred, mask_hr, name, variable))

    out = pd.DataFrame(rows).sort_values(["variable", "model"]).reset_index(drop=True)
    cols = [
        "model",
        "variable",
        "CRPS",
        "RALSD",
        "FSS",
        "SSIM",
        "RMSE",
        "MAE",
        "PITD",
    ]
    out.to_csv(PAPER_STATS_DIR / "SR_metrics_cobweb.csv", index=False, columns=cols)


if __name__ == "__main__":
    main()