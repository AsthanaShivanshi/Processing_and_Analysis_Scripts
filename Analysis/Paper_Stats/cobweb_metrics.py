from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from joblib import Parallel, delayed
from properscoring import crps_ensemble
from skimage.metrics import structural_similarity
from tqdm.auto import tqdm as tqdm_auto


from pysteps.utils.spectral import rapsd

sns.set_style("whitegrid")

SCRIPT_PATH = Path(__file__).resolve()
PAPER_STATS_DIR = SCRIPT_PATH.parent
ANALYSIS_DIR = PAPER_STATS_DIR.parent
PROCESSING_ROOT = ANALYSIS_DIR.parent
PROJECT_ROOT = PROCESSING_ROOT.parent

SAMPLE_DIMS = ("member", "sample", "samples")
EPS = 1e-30


def _clip_nonneg_precip(da: xr.DataArray) -> xr.DataArray:
    return da.clip(min=0)


def _resolve_sample_dim(da: xr.DataArray) -> str | None:
    for d in SAMPLE_DIMS:
        if d in da.dims:
            return d
    return None


def _finite_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _mean_ignore_nan(values) -> float:
    arr = _finite_array(values)
    return float(np.mean(arr)) if arr.size else np.nan


def _summary_from_frame_scores(frame_scores) -> dict[str, float]:
    arr = _finite_array(frame_scores)
    if arr.size == 0:
        return {"mean": np.nan, "min": np.nan, "max": np.nan}
    return {"mean": float(np.mean(arr)), "min": np.nan, "max": np.nan}


def _summary_from_sample_frame_scores(sample_frame_scores: np.ndarray) -> dict[str, float]:
    arr = np.asarray(sample_frame_scores, dtype=float)

    if arr.ndim == 1:
        return _summary_from_frame_scores(arr)
    if arr.ndim != 2:
        raise ValueError("sample_frame_scores must have shape (sample, frame)")

    frame_means = np.array([_mean_ignore_nan(arr[:, t]) for t in range(arr.shape[1])], dtype=float)
    frame_mins = np.array([np.nanmin(arr[:, t]) if np.isfinite(arr[:, t]).any() else np.nan for t in range(arr.shape[1])], dtype=float)
    frame_maxs = np.array([np.nanmax(arr[:, t]) if np.isfinite(arr[:, t]).any() else np.nan for t in range(arr.shape[1])], dtype=float)

    return {
        "mean": _mean_ignore_nan(frame_means),
        "min": _mean_ignore_nan(frame_mins),
        "max": _mean_ignore_nan(frame_maxs),
    }


def _load_field(path: Path, var: str, start: str, end: str, mask: xr.DataArray | None = None, clip_precip: bool = False):
    with xr.open_dataset(path) as ds:
        da = ds[var].sel(time=slice(start, end)).load()
    if clip_precip:
        da = _clip_nonneg_precip(da)
    if mask is not None:
        da = da.where(mask)
    return da


def _load_mask(path: Path, var: str = "TabsD") -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        return ds[var].load()


def _spatial_mask(mask2d: xr.DataArray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask2d is None:
        return np.ones(shape, dtype=bool)
    mv = mask2d.values
    return np.isfinite(mv) & (mv != 0)


def _prepare_ensemble(pred: xr.DataArray) -> np.ndarray:
    sample_dim = _resolve_sample_dim(pred)
    if sample_dim is not None:
        return pred.transpose(sample_dim, "time", "N", "E").values
    return pred.expand_dims(sample=[0]).transpose("sample", "time", "N", "E").values


def _sample_summary_metric(metric_fn, obs: xr.DataArray, pred: xr.DataArray, mask2d=None) -> dict[str, float]:
    sample_dim = _resolve_sample_dim(pred)

    if sample_dim is None:
        scores = metric_fn(obs, pred, mask2d=mask2d, return_scores=True)
        return _summary_from_frame_scores(scores)

    sample_frame_scores = []
    for s in range(pred.sizes[sample_dim]):
        pred_s = pred.isel({sample_dim: s}, drop=True)
        sample_frame_scores.append(np.asarray(metric_fn(obs, pred_s, mask2d=mask2d, return_scores=True), dtype=float))

    return _summary_from_sample_frame_scores(np.asarray(sample_frame_scores, dtype=float))


def _ensemble_summary_metric(metric_fn, obs: xr.DataArray, pred: xr.DataArray, mask2d=None, **kwargs) -> dict[str, float]:
    scores = metric_fn(obs, _prepare_ensemble(pred), mask2d=mask2d, return_scores=True, **kwargs)
    return _summary_from_frame_scores(scores)


# -------------------- CRPS -------------------- #

def _crps_for_frame(t, obs_arr, ens_arr, spatial_valid):
    obs_frame = obs_arr[t]
    ens_frame = ens_arr[:, t, :, :]
    valid = spatial_valid & np.isfinite(obs_frame) & np.all(np.isfinite(ens_frame), axis=0)
    if not np.any(valid):
        return np.nan
    return float(np.nanmean(crps_ensemble(obs_frame[valid], ens_frame[:, valid].T)))


def framewise_spatial_crps(obs, ens_pred, mask2d=None, n_jobs=-1, return_scores=False):
    obs_arr = obs.values
    ens_arr = ens_pred.values if hasattr(ens_pred, "values") else ens_pred
    tmax = min(obs_arr.shape[0], ens_arr.shape[1])
    spatial_valid = _spatial_mask(mask2d, obs_arr.shape[1:])

    scores = Parallel(n_jobs=n_jobs, prefer="threads", require="sharedmem", batch_size=32)(
        delayed(_crps_for_frame)(t, obs_arr, ens_arr, spatial_valid) for t in range(tmax)
    )
    scores = np.asarray(scores, dtype=float)
    return scores if return_scores else (float(np.nanmean(scores)) if np.isfinite(scores).any() else np.nan)


# -------------------- PITD -------------------- #

def _pitd_for_frame(t, obs_arr, ens_arr, spatial_valid, bins, seed=0):
    obs_frame = obs_arr[t]
    ens_frame = ens_arr[:, t, :, :]
    valid = spatial_valid & np.isfinite(obs_frame) & np.all(np.isfinite(ens_frame), axis=0)
    if not np.any(valid):
        return np.nan

    obs_valid = obs_frame[valid]
    ens_valid = ens_frame[:, valid]

    less_than = np.mean(ens_valid < obs_valid[None, :], axis=0)
    equal_to = np.mean(np.isclose(ens_valid, obs_valid[None, :], rtol=0.0, atol=1e-6), axis=0)

    rng = np.random.default_rng(seed + t)
    pit = less_than + rng.random(size=obs_valid.shape) * equal_to

    hist, _ = np.histogram(pit, bins=np.linspace(0, 1, bins + 1))
    if hist.sum() == 0:
        return np.nan

    pit_prob = hist / hist.sum()
    uniform_prob = np.ones(bins) / bins
    return float(np.sqrt(np.mean((pit_prob - uniform_prob) ** 2)))


def framewise_spatial_pitd(obs, ens_pred, bins=20, mask2d=None, n_jobs=-1, return_scores=False, seed=0):
    obs_arr = obs.values
    ens_arr = ens_pred.values if hasattr(ens_pred, "values") else ens_pred
    if ens_arr.ndim != 4:
        raise ValueError("ens_pred must have shape (sample, time, N, E)")

    tmax = min(obs_arr.shape[0], ens_arr.shape[1])
    spatial_valid = _spatial_mask(mask2d, obs_arr.shape[1:])

    scores = Parallel(n_jobs=n_jobs, prefer="threads", require="sharedmem", batch_size=32)(
        delayed(_pitd_for_frame)(t, obs_arr, ens_arr, spatial_valid, bins, seed) for t in range(tmax)
    )
    scores = np.asarray(scores, dtype=float)
    return scores if return_scores else (float(np.nanmean(scores)) if np.isfinite(scores).any() else np.nan)


# -------------------- RALSD -------------------- #

def _frame_rapsd(frame_2d: np.ndarray, mask2d: xr.DataArray | None = None) -> np.ndarray:
    arr = np.asarray(frame_2d, dtype=float)

    if mask2d is not None:
        mv = mask2d.values
        arr = np.where(np.isfinite(mv) & (mv != 0), arr, np.nan)

    arr = np.nan_to_num(arr, nan=0.0)
    return rapsd(arr, fft_method=np.fft, normalize=False) #Shape plus mag.  True : only shape. 


def _ralsd_from_frames(obs_frame: np.ndarray, pred_frame: np.ndarray, mask2d: xr.DataArray | None = None) -> float:
    obs_psd = _frame_rapsd(obs_frame, mask2d=mask2d)
    pred_psd = _frame_rapsd(pred_frame, mask2d=mask2d)

    n = min(obs_psd.size, pred_psd.size)
    if n == 0:
        return np.nan

    ratio = obs_psd[:n] / np.maximum(pred_psd[:n], EPS)
    log_ratio = 10.0 * np.log10(np.maximum(ratio, EPS))
    return float(np.mean(log_ratio**2))



def framewise_spatial_ralsd(obs, pred, mask2d=None, return_scores=False):
    obs, pred = xr.align(obs, pred, join="inner")
    tmax = min(obs.sizes.get("time", 0), pred.sizes.get("time", 0))
    if tmax == 0:
        return np.array([], dtype=float) if return_scores else np.nan

    scores = np.asarray(
        [_ralsd_from_frames(obs.isel(time=t).values, pred.isel(time=t).values, mask2d=mask2d) for t in range(tmax)],
        dtype=float,
    )
    return scores if return_scores else (float(np.nanmean(scores)) if np.isfinite(scores).any() else np.nan)


# -------------------- RMSE -------------------- #

def framewise_rmse(obs, pred, mask2d=None, return_scores=False):
    obs, pred = xr.align(obs, pred, join="inner")
    tmax = min(obs.sizes.get("time", 0), pred.sizes.get("time", 0))
    spatial_valid = _spatial_mask(mask2d, obs.isel(time=0).shape if tmax > 0 else pred.isel(time=0).shape)

    scores = []
    for t in range(tmax):
        o = obs.isel(time=t).values
        p = pred.isel(time=t).values
        valid = spatial_valid & np.isfinite(o) & np.isfinite(p)
        scores.append(np.sqrt(np.mean((p[valid] - o[valid]) ** 2)) if np.any(valid) else np.nan)

    scores = np.asarray(scores, dtype=float)
    return scores if return_scores else (float(np.nanmean(scores)) if np.isfinite(scores).any() else np.nan)


# -------------------- MAE -------------------- #

def framewise_mae(obs, pred, mask2d=None, return_scores=False):
    obs, pred = xr.align(obs, pred, join="inner")
    tmax = min(obs.sizes.get("time", 0), pred.sizes.get("time", 0))
    spatial_valid = _spatial_mask(mask2d, obs.isel(time=0).shape if tmax > 0 else pred.isel(time=0).shape)

    scores = []
    for t in range(tmax):
        o = obs.isel(time=t).values
        p = pred.isel(time=t).values
        valid = spatial_valid & np.isfinite(o) & np.isfinite(p)
        scores.append(np.mean(np.abs(p[valid] - o[valid])) if np.any(valid) else np.nan)

    scores = np.asarray(scores, dtype=float)
    return scores if return_scores else (float(np.nanmean(scores)) if np.isfinite(scores).any() else np.nan)


# -------------------- SSIM -------------------- #

def framewise_ssim(obs, pred, mask2d=None, return_scores=False):
    obs, pred = xr.align(obs, pred, join="inner")
    tmax = min(obs.sizes.get("time", 0), pred.sizes.get("time", 0))

    spatial_mask = None
    if mask2d is not None:
        mv = mask2d.values
        spatial_mask = np.isfinite(mv) & (mv != 0)

    scores = []
    for t in range(tmax):
        o = obs.isel(time=t).values
        p = pred.isel(time=t).values
        finite = np.isfinite(o) & np.isfinite(p)
        valid = finite if spatial_mask is None else (spatial_mask & finite)

        if not np.any(valid):
            scores.append(np.nan)
            continue

        o_fill = np.nanmean(o[valid])
        p_fill = np.nanmean(p[valid])
        o_filled = np.where(valid, o, o_fill)
        p_filled = np.where(valid, p, p_fill)

        data_range = o_filled[valid].max() - o_filled[valid].min()
        if data_range == 0:
            scores.append(1.0 if np.allclose(o_filled, p_filled, equal_nan=True) else 0.0)
            continue

        try:
            scores.append(structural_similarity(o_filled, p_filled, data_range=data_range))
        except Exception:
            scores.append(np.nan)

    scores = np.asarray(scores, dtype=float)
    return scores if return_scores else (float(np.nanmean(scores)) if np.isfinite(scores).any() else np.nan)








#######

def main():
    test_temp = _load_field(
        PROCESSING_ROOT / "data_1971_2023/HR_files_full/TabsD_1971_2023.nc",
        "TabsD",
        "2015-01-01",
        "2023-12-31",
    )
    test_precip = _load_field(
        PROCESSING_ROOT / "data_1971_2023/HR_files_full/RhiresD_1971_2023.nc",
        "RhiresD",
        "2015-01-01",
        "2023-12-31",
        clip_precip=True,
    )

    Swiss_Mask_LR = _load_mask(
        PROJECT_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_LR.nc"
    )
    Swiss_Mask_HR = _load_mask(
        PROJECT_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc"
    ).load()

    test_coarse_temp = _load_field(
        PROJECT_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step2_coarse.nc",
        "TabsD",
        "2015-01-01",
        "2023-12-31",
        mask=Swiss_Mask_LR,
    )
    test_coarse_precip = _load_field(
        PROJECT_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step2_coarse.nc",
        "RhiresD",
        "2015-01-01",
        "2023-12-31",
        mask=Swiss_Mask_LR,
        clip_precip=True,
    )

    models_temp = {
        "Coarse": test_coarse_temp.interp_like(Swiss_Mask_HR, method="nearest"),
        "Bicubic": _load_field(
            PROJECT_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step3_interp_bicubic.nc",
            "TabsD",
            "2015-01-01",
            "2023-12-31",
            mask=Swiss_Mask_HR,
        ),
        "Bilinear": _load_field(
            PROJECT_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step3_interp_bilinear.nc",
            "TabsD",
            "2015-01-01",
            "2023-12-31",
            mask=Swiss_Mask_HR,
        ),
        "UNet": _load_field(
            PROJECT_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/unet_downscaled_test_set_2015_2023.nc",
            "temp",
            "2015-01-01",
            "2023-12-31",
            mask=Swiss_Mask_HR,
        ),
        "DDIM": _load_field(
            PROJECT_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc",
            "temp",
            "2015-01-01",
            "2023-12-31",
            mask=Swiss_Mask_HR,
        ),
        "CFM": _load_field(
            PROJECT_ROOT / "Downscaling_Models/FM_conditional_derived/output_inference/fm_downscaled_test_set_allframes_steps10_samples10.nc",
            "temp",
            "2015-01-01",
            "2023-12-31",
            mask=Swiss_Mask_HR,
        ),
    }

    models_precip = {
        "Coarse": test_coarse_precip.interp_like(Swiss_Mask_HR, method="nearest"),
        "Bicubic": _load_field(
            PROJECT_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp_bicubic.nc",
            "RhiresD",
            "2015-01-01",
            "2023-12-31",
            mask=Swiss_Mask_HR,
            clip_precip=True,
        ),
        "Bilinear": _load_field(
            PROJECT_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp_bilinear.nc",
            "RhiresD",
            "2015-01-01",
            "2023-12-31",
            mask=Swiss_Mask_HR,
            clip_precip=True,
        ),
        "UNet": _load_field(
            PROJECT_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/unet_downscaled_test_set_2015_2023.nc",
            "precip",
            "2015-01-01",
            "2023-12-31",
            mask=Swiss_Mask_HR,
            clip_precip=True,
        ),
        "DDIM": _load_field(
            PROJECT_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc",
            "precip",
            "2015-01-01",
            "2023-12-31",
            mask=Swiss_Mask_HR,
            clip_precip=True,
        ),
        "CFM": _load_field(
            PROJECT_ROOT / "Downscaling_Models/FM_conditional_derived/output_inference/fm_downscaled_test_set_allframes_steps10_samples10.nc",
            "precip",
            "2015-01-01",
            "2023-12-31",
            mask=Swiss_Mask_HR,
            clip_precip=True,
        ),
    }

    test_temp = test_temp.load()
    test_precip = test_precip.load()

    rows = []

    def _metric_bundle(obs, pred, is_ens=False):
        if is_ens:
            crps = _ensemble_summary_metric(
                framewise_spatial_crps, obs, pred, mask2d=Swiss_Mask_HR, n_jobs=-1
            )
            pitd = _ensemble_summary_metric(
                framewise_spatial_pitd, obs, pred, mask2d=Swiss_Mask_HR, bins=20, n_jobs=-1
            )
        else:
            crps = {"mean": np.nan, "min": np.nan, "max": np.nan}
            pitd = {"mean": np.nan, "min": np.nan, "max": np.nan}

        ralsd = _sample_summary_metric(framewise_spatial_ralsd, obs, pred, mask2d=Swiss_Mask_HR)
        ssim = _sample_summary_metric(framewise_ssim, obs, pred, mask2d=Swiss_Mask_HR)
        rmse = _sample_summary_metric(framewise_rmse, obs, pred, mask2d=Swiss_Mask_HR)
        mae = _sample_summary_metric(framewise_mae, obs, pred, mask2d=Swiss_Mask_HR)

        return crps, pitd, ralsd, ssim, rmse, mae

    for name, pred in tqdm_auto(models_temp.items(), desc="Processing temp"):
        is_ens = _resolve_sample_dim(pred) is not None
        crps, pitd, ralsd, ssim, rmse, mae = _metric_bundle(test_temp, pred, is_ens=is_ens)

        rows.append(
            {
                "model": name,
                "variable": "temp",
                "CRPS_mean": crps["mean"],
                "CRPS_min": np.nan,
                "CRPS_max": np.nan,
                "RALSD_mean": ralsd["mean"],
                "RALSD_min": ralsd["min"],
                "RALSD_max": ralsd["max"],
                "SSIM_mean": ssim["mean"],
                "SSIM_min": ssim["min"],
                "SSIM_max": ssim["max"],
                "RMSE_mean": rmse["mean"],
                "RMSE_min": rmse["min"],
                "RMSE_max": rmse["max"],
                "MAE_mean": mae["mean"],
                "MAE_min": mae["min"],
                "MAE_max": mae["max"],
                "PITD_mean": pitd["mean"],
                "PITD_min": np.nan,
                "PITD_max": np.nan,
            }
        )

    for name, pred in tqdm_auto(models_precip.items(), desc="Processing precip"):
        is_ens = _resolve_sample_dim(pred) is not None
        crps, pitd, ralsd, ssim, rmse, mae = _metric_bundle(test_precip, pred, is_ens=is_ens)

        rows.append(
            {
                "model": name,
                "variable": "precip",
                "CRPS_mean": crps["mean"],
                "CRPS_min": np.nan,
                "CRPS_max": np.nan,
                "RALSD_mean": ralsd["mean"],
                "RALSD_min": ralsd["min"],
                "RALSD_max": ralsd["max"],
                "SSIM_mean": ssim["mean"],
                "SSIM_min": ssim["min"],
                "SSIM_max": ssim["max"],
                "RMSE_mean": rmse["mean"],
                "RMSE_min": rmse["min"],
                "RMSE_max": rmse["max"],
                "MAE_mean": mae["mean"],
                "MAE_min": mae["min"],
                "MAE_max": mae["max"],
                "PITD_mean": pitd["mean"],
                "PITD_min": np.nan,
                "PITD_max": np.nan,
            }
        )

    pd.DataFrame(rows).to_csv(PAPER_STATS_DIR / "SR_metrics_cobweb_revised.csv", index=False)

if __name__ == "__main__":
    main()