import pandas as pd
import xarray as xr
import numpy as np
from properscoring import crps_ensemble

from pathlib import Path
from joblib import Parallel, delayed
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


def _get_valid_cells(mask, N, E):
    if mask is None:
        ii, jj = np.indices((N, E))
        return np.column_stack((ii.ravel(), jj.ravel()))
    m = mask.values
    valid = np.isfinite(m) & (m != 0)
    return np.argwhere(valid)


# ----------------------------------------------- #
# CRPS

def _crps_for_grid(i, j, obs_arr, ens_arr):
    obs_series = obs_arr[:, i, j]
    ens_series = ens_arr[:, :, i, j]  # (sample, time)

    valid_time = np.isfinite(obs_series) & np.all(np.isfinite(ens_series), axis=0)
    if np.sum(valid_time) < 2:
        return np.nan

    obs_valid = obs_series[valid_time]
    ens_valid = ens_series[:, valid_time]
    return np.nanmean(crps_ensemble(obs_valid, ens_valid.T))


def gridwise_temporal_crps(obs, ens_pred, mask=None, n_jobs=-1):
    obs_arr = obs.values
    ens_arr = ens_pred.values if hasattr(ens_pred, "values") else ens_pred  # (sample, T, N, E)

    _, N, E = obs_arr.shape
    cells = _get_valid_cells(mask, N, E)

    results = Parallel(n_jobs=n_jobs, prefer="threads", require="sharedmem", batch_size=64)(
        delayed(_crps_for_grid)(i, j, obs_arr, ens_arr) for i, j in cells
    )

    crps_grid = np.full((N, E), np.nan, dtype=np.float64)
    for (i, j), v in zip(cells, results):
        crps_grid[i, j] = v
    return crps_grid


# ----------------------------------------------- #
# LSD

def _lsd_for_grid(i, j, obs_arr, pred_arr, n_fft=256, eps=1e-8):
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

    cells = _get_valid_cells(mask, N, E)

    results = Parallel(n_jobs=n_jobs, prefer="threads", require="sharedmem", batch_size=64)(
        delayed(_lsd_for_grid)(i, j, obs_arr, pred_arr, n_fft, eps) for i, j in cells
    )

    lsd_grid = np.full((N, E), np.nan, dtype=np.float64)
    for (i, j), v in zip(cells, results):
        lsd_grid[i, j] = v

    return lsd_grid


# ----------------------------------------------- #

def spatial_mean_ts(da, mask=None):
    if mask is not None:
        da = da.where(mask)
    return da.mean(dim=["N", "E"], skipna=True)


def rmse(a, b):
    diff = a - b
    rmse_grid = np.sqrt((diff ** 2).mean(dim="time", skipna=True))
    return rmse_grid.mean(dim=["N", "E"], skipna=True).item()


def mae(predictions, targets):
    diff = predictions - targets
    arr = diff.values if hasattr(diff, "values") else diff
    return float(np.nanmean(np.abs(arr)))


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
# PITD

def _pitd_for_grid(i, j, obs_arr, ens_arr, bins):
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

    pit_counts, _ = np.histogram(pit, bins=np.linspace(0, 1, bins + 1))
    if pit_counts.sum() == 0:
        return np.nan

    pit_prob = pit_counts / pit_counts.sum()
    uniform_prob = np.ones(bins) / bins
    return np.sqrt(np.mean((pit_prob - uniform_prob) ** 2))


def gridwise_temporal_pitd(obs, ens_pred, bins=20, n_jobs=-1, mask=None):
    obs_arr = obs.values
    ens_arr = ens_pred.values if hasattr(ens_pred, "values") else ens_pred
    _, N, E = obs_arr.shape

    cells = _get_valid_cells(mask, N, E)

    results = Parallel(n_jobs=n_jobs, prefer="threads", require="sharedmem", batch_size=64)(
        delayed(_pitd_for_grid)(i, j, obs_arr, ens_arr, bins) for i, j in cells
    )

    pitd_grid = np.full((N, E), np.nan, dtype=np.float64)
    for (i, j), v in zip(cells, results):
        pitd_grid[i, j] = v

    return pitd_grid


# ----------------------------------------------- #
# SSIM

def framewise_ssim(obs, pred, mask2d=None):
    ssim_frames = []
    spatial_mask = None
    if mask2d is not None:
        mv = mask2d.values
        spatial_mask = np.isfinite(mv) & (mv != 0)

    for t in range(obs.shape[0]):
        obs_frame = obs.isel(time=t).values
        pred_frame = pred.isel(time=t).values

        finite_mask = np.isfinite(obs_frame) & np.isfinite(pred_frame)
        valid_mask = finite_mask if spatial_mask is None else (spatial_mask & finite_mask)

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
            ssim_val = structural_similarity(obs_filled, pred_filled, data_range=data_range)
        except Exception:
            ssim_val = np.nan

        ssim_frames.append(ssim_val)

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


def _load_all_models(models_dict):
    for k in models_dict:
        models_dict[k] = models_dict[k].load()


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

# Bicubic files
test_temp_bicubic = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step3_interp_bicubic.nc")["TabsD"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

test_precip_bicubic = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp_bicubic.nc")["RhiresD"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

# UNet
test_temp_unet = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/unet_downscaled_test_set_2015_2023.nc")["temp"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

test_precip_unet = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/unet_downscaled_test_set_2015_2023.nc")["precip"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

# DDIM files
test_temp_ddim = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc")["temp"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

test_precip_ddim = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/DDIM_conditional_derived/output_inference/ddim_downscaled_test_set_S30_samples10_eta0.0.nc")["precip"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

# FM files
test_temp_cfm = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/FM_conditional_derived/output_inference/fm_downscaled_test_set_allframes_steps10_samples10.nc")["temp"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

test_precip_cfm = (
    xr.open_dataset(PROJECT_ROOT / "Downscaling_Models/FM_conditional_derived/output_inference/fm_downscaled_test_set_allframes_steps10_samples10.nc")["precip"]
    .sel(time=slice("2015-01-01", "2023-12-31"))
).where(Swiss_Mask_HR)

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

# Force one-time memory load (avoid repeated lazy I/O during metric loops)
Swiss_Mask_HR = Swiss_Mask_HR.load()
test_temp = test_temp.load()
test_precip = test_precip.load()
_load_all_models(models_temp)
_load_all_models(models_precip)

# Precompute obs spatial means once
test_temp_spatial_mean = spatial_mean_ts(test_temp, Swiss_Mask_HR)
test_precip_spatial_mean = spatial_mean_ts(test_precip, Swiss_Mask_HR)

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

    crps_grid = gridwise_temporal_crps(test_temp, pred_ens, mask=Swiss_Mask_HR, n_jobs=-1)
    pitd_grid = gridwise_temporal_pitd(test_temp, pred_ens, bins=20, mask=Swiss_Mask_HR, n_jobs=-1)
    lsd_grid = gridwise_temporal_lsd(test_temp, pred_med, mask=Swiss_Mask_HR, n_jobs=-1)
    ssim_val = framewise_ssim(test_temp, pred_med, mask2d=Swiss_Mask_HR)
    rmse_val = rmse(test_temp, pred_med)

    pred_spatial_mean = spatial_mean_ts(pred_med, Swiss_Mask_HR)
    mae_val = mae(pred_spatial_mean, test_temp_spatial_mean)
    psnr_val = psnr(pred_spatial_mean, test_temp_spatial_mean)

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

    crps_grid = gridwise_temporal_crps(test_precip, pred_ens, mask=Swiss_Mask_HR, n_jobs=-1)
    pitd_grid = gridwise_temporal_pitd(test_precip, pred_ens, bins=20, mask=Swiss_Mask_HR, n_jobs=-1)
    lsd_grid = gridwise_temporal_lsd(test_precip, pred_med, mask=Swiss_Mask_HR, n_jobs=-1)
    ssim_val = framewise_ssim(test_precip, pred_med, mask2d=Swiss_Mask_HR)
    rmse_val = rmse(test_precip, pred_med)

    pred_spatial_mean = spatial_mean_ts(pred_med, Swiss_Mask_HR)
    mae_val = mae(pred_spatial_mean, test_precip_spatial_mean)
    psnr_val = psnr(pred_spatial_mean, test_precip_spatial_mean)

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