from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from properscoring import crps_ensemble
from pysteps.utils.spectral import rapsd
from skimage.metrics import structural_similarity

sns.set_style("whitegrid")

SCRIPT_PATH = Path(__file__).resolve()
PAPER_STATS_DIR = SCRIPT_PATH.parent
PROCESSING_ROOT = PAPER_STATS_DIR.parent.parent.parent

START = "2015-01-01"
END = "2023-12-31"
SAMPLE_DIMS = ("member", "sample", "samples")
EPS = 1e-30


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
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


def _mean(x) -> float:
    a = _finite(x)
    return float(a.mean()) if a.size else np.nan


def _frame_to_ensemble(pred_frame: xr.DataArray) -> np.ndarray:
    sd = _sample_dim(pred_frame)
    if sd is None:
        spatial = list(pred_frame.dims)
        return pred_frame.transpose(*spatial).values[None, ...]
    spatial = [d for d in pred_frame.dims if d != sd]
    return pred_frame.transpose(sd, *spatial).values


def _ralsd_2d(obs2d: np.ndarray, pred2d: np.ndarray, mask2d: xr.DataArray | None = None) -> float:
    valid = np.isfinite(obs2d) & np.isfinite(pred2d)
    if mask2d is not None:
        valid &= _spatial_mask(mask2d, obs2d.shape)
    if not np.any(valid):
        return np.nan

    o = np.where(valid, obs2d, np.nan)
    p = np.where(valid, pred2d, np.nan)
    o = np.nan_to_num(o, nan=0.0) #Setting to 0 for FFT,, 
    p = np.nan_to_num(p, nan=0.0)

    o_psd = rapsd(o, fft_method=np.fft, normalize=False)
    p_psd = rapsd(p, fft_method=np.fft, normalize=False)

    n = min(o_psd.size, p_psd.size)


    if n == 0:
        return np.nan


    o_psd = o_psd[:n]
    p_psd = p_psd[:n]

    o_sum = float(np.sum(o_psd))
    p_sum = float(np.sum(p_psd))


    if o_sum <= 0 or p_sum <= 0:
        return np.nan

    o_psd = o_psd / o_sum #Spctral ration. :: from ML Cordex. 
    p_psd = p_psd / p_sum

    ratio = o_psd / np.maximum(p_psd, EPS)
    return float(np.mean((10.0 * np.log10(np.maximum(ratio, EPS))) ** 2)) #MSE- 


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

    obs = obs2d[valid]
    ens = ens3d[:, valid].T
    return float(np.mean(crps_ensemble(obs, ens)))


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

    rng = np.random.default_rng(seed)
    pit = less + rng.random(obs.size) * equal

    hist, _ = np.histogram(pit, bins=np.linspace(0, 1, bins + 1))
    if hist.sum() == 0:
        return np.nan

    p = hist / hist.sum()
    u = np.ones(bins) / bins
    return float(np.sqrt(np.mean((p - u) ** 2))) #RMSE used for PITD. 


def _series_mean(obs: xr.DataArray, pred: xr.DataArray, fn, mask2d=None) -> float:
    sd = _sample_dim(pred)
    if sd is not None:
        pred = pred.mean(dim=sd, skipna=True)
    obs, pred = xr.align(obs, pred, join="inner")
    tmax = min(obs.sizes.get("time", 0), pred.sizes.get("time", 0))
    scores = [fn(obs.isel(time=t).values, pred.isel(time=t).values, mask2d) for t in range(tmax)]
    return _mean(scores)


def _series_bounds(obs: xr.DataArray, pred: xr.DataArray, fn, mask2d=None) -> tuple[float, float]:
    sd = _sample_dim(pred)
    if sd is None:
        return np.nan, np.nan

    obs, pred = xr.align(obs, pred, join="inner")
    tmax = min(obs.sizes.get("time", 0), pred.sizes.get("time", 0))
    mins, maxs = [], []

    for t in range(tmax):
        o = obs.isel(time=t).values
        vals = [
            fn(o, pred.isel({sd: s, "time": t}).values, mask2d)
            for s in range(pred.sizes[sd])
        ]
        vals = _finite(vals)
        mins.append(np.min(vals) if vals.size else np.nan)
        maxs.append(np.max(vals) if vals.size else np.nan)

    return _mean(mins), _mean(maxs)


def _series_crps(obs: xr.DataArray, pred: xr.DataArray, mask2d=None) -> float:
    obs, pred = xr.align(obs, pred, join="inner")
    tmax = min(obs.sizes.get("time", 0), pred.sizes.get("time", 0))
    scores = []
    for t in range(tmax):
        scores.append(_crps_frame(obs.isel(time=t).values, pred.isel(time=t), mask2d))
    return _mean(scores)


def _series_pitd(obs: xr.DataArray, pred: xr.DataArray, mask2d=None, bins: int = 20) -> float:
    obs, pred = xr.align(obs, pred, join="inner")
    tmax = min(obs.sizes.get("time", 0), pred.sizes.get("time", 0))
    scores = []
    for t in range(tmax):
        scores.append(_pitd_frame(obs.isel(time=t).values, pred.isel(time=t), mask2d, bins=bins, seed=t))
    return _mean(scores)


def _row(obs: xr.DataArray, pred: xr.DataArray, mask2d: xr.DataArray, model: str, variable: str) -> dict[str, float | str]:
    crps = _series_crps(obs, pred, mask2d)
    pitd = _series_pitd(obs, pred, mask2d)

    ralsd = _series_mean(obs, pred, _ralsd_2d, mask2d)
    ssim = _series_mean(obs, pred, _ssim_2d, mask2d)
    rmse = _series_mean(obs, pred, _rmse_2d, mask2d)
    mae = _series_mean(obs, pred, _mae_2d, mask2d)

    ralsd_min, ralsd_max = _series_bounds(obs, pred, _ralsd_2d, mask2d)
    ssim_min, ssim_max = _series_bounds(obs, pred, _ssim_2d, mask2d)
    rmse_min, rmse_max = _series_bounds(obs, pred, _rmse_2d, mask2d)
    mae_min, mae_max = _series_bounds(obs, pred, _mae_2d, mask2d)

    return {
        "model": model,
        "variable": variable,
        "CRPS_mean": crps,
        "RALSD_mean": ralsd,
        "RALSD_min": ralsd_min,
        "RALSD_max": ralsd_max,
        "SSIM_mean": ssim,
        "SSIM_min": ssim_min,
        "SSIM_max": ssim_max,
        "RMSE_mean": rmse,
        "RMSE_min": rmse_min,
        "RMSE_max": rmse_max,
        "MAE_mean": mae,
        "MAE_min": mae_min,
        "MAE_max": mae_max,
        "PITD_mean": pitd,
    }


def main() -> None:
    mask_lr = _load_mask(PROCESSING_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_LR.nc", "TabsD")
    mask_hr = _load_mask(PROCESSING_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc", "TabsD")

    obs_temp = _load_field(PROCESSING_ROOT / "Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/TabsD_1971_2023.nc", "TabsD")
    obs_precip = _load_field(PROCESSING_ROOT / "Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/RhiresD_1971_2023.nc", "RhiresD", clip=True)

    coarse_temp = _load_field(PROCESSING_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step2_coarse.nc", "TabsD", mask_lr).interp_like(mask_hr, method="nearest")
    coarse_precip = _load_field(PROCESSING_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step2_coarse.nc", "RhiresD", mask_lr, clip=True).interp_like(mask_hr, method="nearest")




    temp_models = {
        "Coarse": coarse_temp,
        "Bicubic": _load_field(PROCESSING_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step3_interp_bicubic.nc", "TabsD", mask_hr),
        "Bilinear": _load_field(PROCESSING_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step3_interp_bilinear.nc", "TabsD", mask_hr),
        "UNet": _load_field(PROCESSING_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/unet_downscaled_test_set_2015_2023.nc", "temp", mask_hr),
        "DDIM": _load_field(PROCESSING_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc", "temp", mask_hr),
        "CFM": _load_field(PROCESSING_ROOT / "Downscaling_Models/FM_conditional_derived/output_inference/fm_downscaled_test_set_allframes_steps10_samples10.nc", "temp", mask_hr),
    }



    precip_models = {
        "Coarse": coarse_precip,
        "Bicubic": _load_field(PROCESSING_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp_bicubic.nc", "RhiresD", mask_hr, clip=True),
        "Bilinear": _load_field(PROCESSING_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp_bilinear.nc", "RhiresD", mask_hr, clip=True),
        "UNet": _load_field(PROCESSING_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/unet_downscaled_test_set_2015_2023.nc", "precip", mask_hr, clip=True),
        "DDIM": _load_field(PROCESSING_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc", "precip", mask_hr, clip=True),
        "CFM": _load_field(PROCESSING_ROOT / "Downscaling_Models/FM_conditional_derived/output_inference/fm_downscaled_test_set_allframes_steps10_samples10.nc", "precip", mask_hr, clip=True),
    }



    rows = []
    for name, pred in temp_models.items():
        rows.append(_row(obs_temp, pred, mask_hr, name, "temp"))
    for name, pred in precip_models.items():
        rows.append(_row(obs_precip, pred, mask_hr, name, "precip"))

    cols = [
        "model", "variable",
        "CRPS_mean", 
        "RALSD_mean", "RALSD_min", "RALSD_max",
        "SSIM_mean", "SSIM_min", "SSIM_max",
        "RMSE_mean", "RMSE_min", "RMSE_max",
        "MAE_mean", "MAE_min", "MAE_max",
        "PITD_mean",
    ]

    out = pd.DataFrame(rows, columns=cols).sort_values(["variable", "model"]).reset_index(drop=True)
    out.to_csv(PAPER_STATS_DIR / "SR_metrics_cobweb_revised.csv", index=False)


if __name__ == "__main__":
    main()