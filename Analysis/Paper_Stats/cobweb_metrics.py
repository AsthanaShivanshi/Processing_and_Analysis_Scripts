import pandas as pd
import xarray as xr
import numpy as np
from properscoring import crps_ensemble

from pathlib import Path

from joblib import Parallel, delayed, parallel

from skimage.metrics import structural_similarity

from tqdm.auto import tqdm as tqdm_auto

import seaborn as sns

sns.set_style("whitegrid")


# PWD: PAS

# -------------------------------------------------------------------- #

SCRIPT_PATH = Path(__file__).resolve()
PAPER_STATS_DIR = SCRIPT_PATH.parent
ANALYSIS_DIR = PAPER_STATS_DIR.parent
PROCESSING_ROOT = ANALYSIS_DIR.parent
PROJECT_ROOT = PROCESSING_ROOT.parent


def valid_mask_cell(mask, i, j):
    if mask is None:
        return True
    value = mask.values[i, j]
    return np.isfinite(value) and bool(value)


# ----------------------------------------------- #

def gridwise_temporal_crps(obs, ens_pred, mask=None):
    obs_arr = obs.values
    ens_arr = ens_pred.values if hasattr(ens_pred, "values") else ens_pred  # (sample, T, N, E)

    _, N, E = obs_arr.shape
    crps_grid = np.full((N, E), np.nan)

    for i in range(N):
        for j in range(E):
            if not valid_mask_cell(mask, i, j):
                continue

            obs_series = obs_arr[:, i, j]
            ens_series = ens_arr[:, :, i, j]

            valid_time = np.isfinite(obs_series) & np.all(np.isfinite(ens_series), axis=0)

            if np.sum(valid_time) < 2:
                continue

            obs_valid = obs_series[valid_time]
            ens_valid = ens_series[:, valid_time]

            crps_vals = crps_ensemble(obs_valid, ens_valid.T)
            crps_grid[i, j] = np.nanmean(crps_vals)

    return crps_grid


# ----------------------------------------------- #

def lsd_for_grid(i, j, obs_arr, pred_arr, n_fft=256, eps=1e-8):
    obs_series = obs_arr[:, i, j]
    pred_series = pred_arr[:, i, j]

    valid_time = np.isfinite(obs_series) & np.isfinite(pred_series)

    if np.sum(valid_time) < n_fft:
        return np.nan

    obs_valid = obs_series[valid_time]
    pred_valid = pred_series[valid_time]

    obs_fft = np.fft.rfft(obs_valid, n=n_fft)
    pred_fft = np.fft.rfft(pred_valid, n=n_fft)

    obs_log = np.log(np.abs(obs_fft) + eps)
    pred_log = np.log(np.abs(pred_fft) + eps)

    return np.sqrt(np.mean((obs_log - pred_log) ** 2))


def gridwise_temporal_lsd(obs, pred, n_fft=256, eps=1e-8, n_jobs=-1, mask=None):
    obs_arr = obs.values
    pred_arr = pred.values
    _, N, E = obs_arr.shape

    tasks = []
    for i in range(N):
        for j in range(E):
            if not valid_mask_cell(mask, i, j):
                continue
            tasks.append((i, j))

    lsd_grid = np.full((N, E), np.nan)
    original_callback = parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallback(original_callback):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    try:
        parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

        with tqdm_auto(total=len(tasks), desc="Grid cells processed", unit="cell") as tqdm_object:
            results = Parallel(n_jobs=n_jobs)(
                delayed(lsd_for_grid)(i, j, obs_arr, pred_arr, n_fft, eps)
                for i, j in tasks
            )
    finally:
        parallel.BatchCompletionCallBack = original_callback

    for idx, (i, j) in enumerate(tasks):
        lsd_grid[i, j] = results[idx]

    return lsd_grid


# ----------------------------------------------- #

def spatial_mean_ts(da, mask=None):
    if mask is not None:
        da = da.where(mask)
    return da.mean(dim=["N", "E"], skipna=True)


def rmse(a, b):
    diff = a - b
    rmse_grid = np.sqrt((diff ** 2).mean(dim="time", skipna=True))
    spatial_mean_rmse = rmse_grid.mean(dim=["N", "E"], skipna=True).item()
    return spatial_mean_rmse


# ----------------------------------------------- #

def mae(predictions, targets):
    diff = predictions - targets
    return float(np.nanmean(np.abs(diff.values if hasattr(diff, "values") else diff)))


# ----------------------------------------------- #

def psnr(predictions, targets):
    diff = predictions - targets
    diff_values = diff.values if hasattr(diff, "values") else diff
    target_values = targets.values if hasattr(targets, "values") else targets

    mse = float(np.nanmean(diff_values ** 2))

    if mse == 0:
        return np.inf

    max_pixel_value = float(np.nanmax(target_values))

    if not np.isfinite(max_pixel_value) or max_pixel_value == 0:
        return np.nan

    return float(20 * np.log10(max_pixel_value / np.sqrt(mse)))


# ----------------------------------------------- #

def pitd_for_grid(i, j, obs_arr, ens_arr, bins):
    obs_series = obs_arr[:, i, j]
    ens_series = ens_arr[:, :, i, j]

    valid_time = np.isfinite(obs_series) & np.all(np.isfinite(ens_series), axis=0)

    if np.sum(valid_time) < 2:
        return np.nan

    obs_valid = obs_series[valid_time]
    ens_valid = ens_series[:, valid_time]

    less_than = np.mean(ens_valid < obs_valid[None, :], axis=0)
    equal_to = np.mean(ens_valid == obs_valid[None, :], axis=0)

    pit = less_than + 0.5 * equal_to

    if pit.size == 0:
        return np.nan

    bin_edges = np.linspace(0, 1, bins + 1)
    pit_counts, _ = np.histogram(pit, bins=bin_edges)

    if pit_counts.sum() == 0:
        return np.nan

    pit_prob = pit_counts / pit_counts.sum()
    uniform_prob = np.ones(bins) / bins

    return np.sqrt(np.mean((pit_prob - uniform_prob) ** 2))


def gridwise_temporal_pitd(obs, ens_pred, bins=20, n_jobs=-1, mask=None):
    obs_arr = obs.values
    ens_arr = ens_pred.values if hasattr(ens_pred, "values") else ens_pred
    _, N, E = obs_arr.shape

    tasks = []
    for i in range(N):
        for j in range(E):
            if not valid_mask_cell(mask, i, j):
                continue
            tasks.append((i, j))

    pitd_grid_values = np.full((N, E), np.nan)

    original_callback = parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallback(original_callback):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    try:
        parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

        with tqdm_auto(total=len(tasks), desc="Grid cells processed", unit="cell") as tqdm_object:
            results = Parallel(n_jobs=n_jobs)(
                delayed(pitd_for_grid)(i, j, obs_arr, ens_arr, bins)
                for i, j in tasks
            )
    finally:
        parallel.BatchCompletionCallBack = original_callback

    for idx, (i, j) in enumerate(tasks):
        pitd_grid_values[i, j] = results[idx]

    return pitd_grid_values


# ----------------------------------------------- #

# SSIM

def framewise_ssim(obs, pred, mask2d=None):
    ssim_frames = []

    for t in range(obs.shape[0]):
        obs_frame = obs.isel(time=t).values
        pred_frame = pred.isel(time=t).values

        finite_mask = np.isfinite(obs_frame) & np.isfinite(pred_frame)

        if mask2d is not None:
            mask_values = mask2d.values
            spatial_mask = np.isfinite(mask_values) & (mask_values != 0)
            valid_mask = spatial_mask & finite_mask
        else:
            valid_mask = finite_mask

        if not np.any(valid_mask):
            ssim_frames.append(np.nan)
            continue

        obs_fill = np.nanmean(obs_frame[valid_mask])
        pred_fill = np.nanmean(pred_frame[valid_mask])

        obs_filled = np.where(valid_mask, obs_frame, obs_fill)
        pred_filled = np.where(valid_mask, pred_frame, pred_fill)

        data_range = obs_filled[valid_mask].max() - obs_filled[valid_mask].min()
        if data_range == 0:
            ssim_frames.append(np.nan)
            continue

        try:
            ssim = structural_similarity(obs_filled, pred_filled, data_range=data_range)
        except Exception:
            ssim = np.nan

        ssim_frames.append(ssim)

    return np.nanmean(ssim_frames)


# ----------------------------------------------- #

def prepare_ensemble_and_median(pred, pred_median=None):
    sample_dim = None
    for d in ("sample", "samples"):
        if d in pred.dims:
            sample_dim = d
            break

    if sample_dim is not None:
        pred_ens = pred.transpose(sample_dim, "time", "N", "E").values
        pred_med = pred_median if pred_median is not None else pred.median(dim=sample_dim, skipna=True)
    else:
        pred_ens = pred.expand_dims(sample=[0]).transpose("sample", "time", "N", "E").values
        pred_med = pred

    return pred_ens, pred_med


# ----------------------------------------------- #

# targets

test_temp = (
    xr.open_dataset(PROCESSING_ROOT / "data_1971_2023/HR_files_full/TabsD_1971_2023.nc")["TabsD"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
)

test_precip = (
    xr.open_dataset(PROCESSING_ROOT / "data_1971_2023/HR_files_full/RhiresD_1971_2023.nc")["RhiresD"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
)


# Masks

Swiss_Mask_LR = xr.open_dataset(
    PROJECT_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_LR.nc"
)["TabsD"]

Swiss_Mask_HR = xr.open_dataset(
    PROJECT_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/Swiss_Mask_HR.nc"
)["TabsD"]


# Coarse

test_coarse_temp = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step2_coarse.nc")["TabsD"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_LR)

test_coarse_precip = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step2_coarse.nc")["RhiresD"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_LR)

# Same grid necessary for image metrics.
test_coarse_temp_interp = test_coarse_temp.interp_like(Swiss_Mask_HR, method="nearest")
test_coarse_precip_interp = test_coarse_precip.interp_like(Swiss_Mask_HR, method="nearest")

# Bilinear files

test_temp_bilinear = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step3_interp_bilinear.nc")["TabsD"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

test_precip_bilinear = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp_bilinear.nc")["RhiresD"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

test_temp_bilinear = test_temp_bilinear.where(Swiss_Mask_HR)
test_precip_bilinear = test_precip_bilinear.where(Swiss_Mask_HR)

# Bicubic files

test_temp_bicubic = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step3_interp_bicubic.nc")["TabsD"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

test_precip_bicubic = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp_bicubic.nc")["RhiresD"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

test_temp_bicubic = test_temp_bicubic.where(Swiss_Mask_HR)
test_precip_bicubic = test_precip_bicubic.where(Swiss_Mask_HR)


# UNet

test_temp_unet = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/unet_downscaled_test_set_2015_2023.nc")["temp"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

test_precip_unet = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/unet_downscaled_test_set_2015_2023.nc")["precip"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

test_temp_unet = test_temp_unet.where(Swiss_Mask_HR)
test_precip_unet = test_precip_unet.where(Swiss_Mask_HR)

# DDIM files

test_temp_ddim = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc")["temp"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

test_precip_ddim = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc")["precip"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

test_temp_ddim = test_temp_ddim.where(Swiss_Mask_HR)
test_precip_ddim = test_precip_ddim.where(Swiss_Mask_HR)

# FM files

test_temp_cfm = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/FM_conditional_derived/output_inference/fm_downscaled_test_set_allframes_steps10_samples10.nc")["temp"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

test_precip_cfm = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/FM_conditional_derived/output_inference/fm_downscaled_test_set_allframes_steps10_samples10.nc")["precip"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

test_temp_cfm = test_temp_cfm.where(Swiss_Mask_HR)
test_precip_cfm = test_precip_cfm.where(Swiss_Mask_HR)

# DDIM median files

test_temp_ddim_median = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0_median.nc")["temp"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

test_precip_ddim_median = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0_median.nc")["precip"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

# FM median files

test_temp_cfm_median = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/FM_conditional_derived/output_inference/fm_downscaled_test_set_allframes_steps10_samples10_median.nc")["temp"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

test_precip_cfm_median = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/FM_conditional_derived/output_inference/fm_downscaled_test_set_allframes_steps10_samples10_median.nc")["precip"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

test_temp_cfm_median = test_temp_cfm_median.where(Swiss_Mask_HR)
test_precip_cfm_median = test_precip_cfm_median.where(Swiss_Mask_HR)

# -------------------------------------------------------------------- #

models_temp = {
    "Coarse": test_coarse_temp_interp,
    "Bicubic": test_temp_bicubic,
    "Bilinear": test_temp_bilinear,
    "UNet": test_temp_unet,
    "DDIM": test_temp_ddim,
    "DDIM_median": test_temp_ddim_median,
    "CFM": test_temp_cfm,
    "CFM_median": test_temp_cfm_median,
}

models_precip = {
    "Coarse": test_coarse_precip_interp,
    "Bicubic": test_precip_bicubic,
    "Bilinear": test_precip_bilinear,
    "UNet": test_precip_unet,
    "DDIM": test_precip_ddim,
    "DDIM_median": test_precip_ddim_median,
    "CFM": test_precip_cfm,
    "CFM_median": test_precip_cfm_median,
}

# -------------------------------------------------------------------- #

metrics = {}

for name, pred in tqdm_auto(models_temp.items(), desc="Processing temp"):
    print(f"Processing {name} for temperature...")

    if name == "DDIM":
        pred_ens, pred_med = prepare_ensemble_and_median(pred, test_temp_ddim_median)
    elif name == "CFM":
        pred_ens, pred_med = prepare_ensemble_and_median(pred, test_temp_cfm_median)
    else:
        pred_ens, pred_med = prepare_ensemble_and_median(pred)

    crps_grid = gridwise_temporal_crps(test_temp, pred_ens, mask=Swiss_Mask_HR)
    pitd_grid = gridwise_temporal_pitd(test_temp, pred_ens, bins=20, mask=Swiss_Mask_HR)
    lsd_grid = gridwise_temporal_lsd(test_temp, pred_med, mask=Swiss_Mask_HR)
    ssim_val = framewise_ssim(test_temp, pred_med, mask2d=Swiss_Mask_HR)
    rmse_val = rmse(test_temp, pred_med)

    mae_val = mae(
        spatial_mean_ts(pred_med, Swiss_Mask_HR),
        spatial_mean_ts(test_temp, Swiss_Mask_HR),
    )
    psnr_val = psnr(
        spatial_mean_ts(pred_med, Swiss_Mask_HR),
        spatial_mean_ts(test_temp, Swiss_Mask_HR),
    )

    metrics[f"{name}_temp"] = {
        "model": name,
        "variable": "temp",
        "CRPS": np.nanmean(crps_grid),
        "LSD_ensmedian": np.nanmean(lsd_grid),
        "SSIM_ensmedian": ssim_val,
        "RMSE_ensmedian": rmse_val,
        "MAE_ensmedian": mae_val,
        "PSNR_ensmedian": psnr_val,
        "PITD": np.nanmean(pitd_grid),
    }

# -------------------------------------------------------------------- #

for name, pred in tqdm_auto(models_precip.items(), desc="Processing precip"):
    print(f"Processing {name} for precipitation...")

    if name == "DDIM":
        pred_ens, pred_med = prepare_ensemble_and_median(pred, test_precip_ddim_median)
    elif name == "CFM":
        pred_ens, pred_med = prepare_ensemble_and_median(pred, test_precip_cfm_median)
    else:
        pred_ens, pred_med = prepare_ensemble_and_median(pred)

    crps_grid = gridwise_temporal_crps(test_precip, pred_ens, mask=Swiss_Mask_HR)
    pitd_grid = gridwise_temporal_pitd(test_precip, pred_ens, bins=20, mask=Swiss_Mask_HR)
    lsd_grid = gridwise_temporal_lsd(test_precip, pred_med, mask=Swiss_Mask_HR)
    ssim_val = framewise_ssim(test_precip, pred_med, mask2d=Swiss_Mask_HR)
    rmse_val = rmse(test_precip, pred_med)

    mae_val = mae(
        spatial_mean_ts(pred_med, Swiss_Mask_HR),
        spatial_mean_ts(test_precip, Swiss_Mask_HR),
    )
    psnr_val = psnr(
        spatial_mean_ts(pred_med, Swiss_Mask_HR),
        spatial_mean_ts(test_precip, Swiss_Mask_HR),
    )

    metrics[f"{name}_precip"] = {
        "model": name,
        "variable": "precip",
        "CRPS": np.nanmean(crps_grid),
        "LSD_ensmedian": np.nanmean(lsd_grid),
        "SSIM_ensmedian": ssim_val,
        "RMSE_ensmedian": rmse_val,
        "MAE_ensmedian": mae_val,
        "PSNR_ensmedian": psnr_val,
        "PITD": np.nanmean(pitd_grid),
    }

metric_df = pd.DataFrame.from_dict(metrics, orient="index")
metric_df.to_csv(PAPER_STATS_DIR / "SR_metrics_cobweb.csv", index=False)