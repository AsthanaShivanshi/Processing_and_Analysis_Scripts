from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from properscoring import crps_ensemble
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity
from tqdm import tqdm

try:
    from scores.categorical.fss import fss as scores_fss
except Exception:
    scores_fss = None

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

# FSS settings

FSS_THRESHOLD_MM_DAY = 1.0  
FSS_WINDOW_KM = 3.0
FSS_WINDOW_PX = max(1, int(round(FSS_WINDOW_KM / GRID_SPACING_KM)))


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


def _time_indices(da: xr.DataArray, valid_times: np.ndarray | None = None) -> list[int | None]:
    if "time" not in da.dims:
        return [None]
    if valid_times is not None:
        return [int(t) for t in valid_times]
    return list(range(da.sizes.get("time", 0)))


def _get_valid_precip_times(obs: xr.DataArray) -> np.ndarray:
    nt = obs.sizes.get("time", 1)
    valid = []
    for t in range(nt):
        frame = obs.isel(time=t).values.astype(float)
        finite = np.isfinite(frame)
        if finite.sum() > 0 and np.nanmean(frame[finite]) >= PRECIP_MIN_MEAN:
            valid.append(t)
    return np.array(valid, dtype=int)


# FSS 


def _fss_fallback(
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

    obs = np.where(valid, obs2d, np.nan)
    pred = np.where(valid, pred2d, np.nan)

    if scores_fss is not None:
        attempts = (
            lambda: scores_fss(fcst=pred, obs=obs, threshold=threshold, window_size=window_px),
            lambda: scores_fss(pred, obs, threshold=threshold, window_size=window_px),
            lambda: scores_fss(fcst=pred, obs=obs, threshold=threshold, neighbourhood_size=window_px),
            lambda: scores_fss(pred, obs, threshold=threshold, neighbourhood_size=window_px),
        )
        for call in attempts:
            try:
                return float(call())
            except Exception:
                pass

    return _fss_fallback(obs2d, pred2d, mask2d, threshold=threshold, window_px=window_px)


def _series_fss(obs: xr.DataArray, pred: xr.DataArray, mask2d=None) -> float:
    obs, pred = xr.align(obs, pred, join="inner")
    sd = _sample_dim(pred)
    vals: list[float] = []

    for t in _time_indices(obs):
        obs_t = obs if t is None else obs.isel(time=t)
        pred_t = pred if t is None else pred.isel(time=t)

        if sd is not None and sd in pred_t.dims:
            for s in range(pred_t.sizes[sd]):
                vals.append(_fss_2d(obs_t.values, pred_t.isel({sd: s}).values, mask2d))
        else:
            vals.append(_fss_2d(obs_t.values, pred_t.values, mask2d))

    return _mean(vals)




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


def _series_framewise(obs: xr.DataArray, pred: xr.DataArray, fn, mask2d=None, is_precip: bool = False) -> float:
    obs, pred = xr.align(obs, pred, join="inner")
    sd = _sample_dim(pred)
    vals: list[float] = []

    valid_times = _get_valid_precip_times(obs) if is_precip else None
    for t in _time_indices(obs, valid_times):
        obs_t = obs if t is None else obs.isel(time=t)
        pred_t = pred if t is None else pred.isel(time=t)

        if sd is not None and sd in pred_t.dims:
            for s in range(pred_t.sizes[sd]):
                vals.append(fn(obs_t.values, pred_t.isel({sd: s}).values, mask2d))
        else:
            vals.append(fn(obs_t.values, pred_t.values, mask2d))

    return _mean(vals)


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

    return {
        "model": model,
        "variable": variable,
        "CRPS": _series_crps(obs, pred, mask2d),
        "SSIM": _series_framewise(obs, pred, _ssim_2d, mask2d),
        "RMSE": _series_framewise(obs, pred, _rmse_2d, mask2d),
        "MAE": _series_framewise(obs, pred, _mae_2d, mask2d),
        "PITD": _series_pitd(obs, pred, mask2d),
        "FSS": _series_fss(obs, pred, mask2d) if is_precip else np.nan,
    }


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
        "FSS",
        "SSIM",
        "RMSE",
        "MAE",
        "PITD",
    ]
    out.to_csv(PAPER_STATS_DIR / "SR_metrics_cobweb.csv", index=False, columns=cols)


if __name__ == "__main__":
    main()