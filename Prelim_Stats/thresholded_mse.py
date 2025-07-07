import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config
import argparse
import os

# Argparsing for running as an array
parser = argparse.ArgumentParser(description="Thresholded MSE Calculation")
parser.add_argument("--var", type=int, required=True, help="Variable index (0-3)")
args = parser.parse_args()

varnames = {
    "precip": "RhiresD",
    "temp": "TabsD",
    "tmin": "TminD",
    "tmax": "TmaxD"
}
var_list = list(varnames.keys())
var = var_list[args.var]
file_var = varnames[var]

unet_pretrain_path = f"{config.UNET_1771_DIR}/Pretraining_Dataset_Downscaled_Predictions_2011_2020.nc"
unet_train_path = f"{config.UNET_1971_DIR}/Training_Dataset_Downscaled_Predictions_2011_2020.nc"
target_files = {
    "RhiresD": f"{config.TARGET_DIR}/RhiresD_1971_2023.nc",
    "TabsD": f"{config.TARGET_DIR}/TabsD_1971_2023.nc",
    "TminD": f"{config.TARGET_DIR}/TminD_1971_2023.nc",
    "TmaxD": f"{config.TARGET_DIR}/TmaxD_1971_2023.nc",
}
bicubic_files = {
    "RhiresD": f"{config.DATASETS_TRAINING_DIR}/RhiresD_step3_interp.nc",
    "TabsD":   f"{config.DATASETS_TRAINING_DIR}/TabsD_step3_interp.nc",
    "TminD":   f"{config.DATASETS_TRAINING_DIR}/TminD_step3_interp.nc",
    "TmaxD":   f"{config.DATASETS_TRAINING_DIR}/TmaxD_step3_interp.nc",
}


unet_pretrain_ds = xr.open_dataset(unet_pretrain_path)
unet_train_ds = xr.open_dataset(unet_train_path)

quantiles = list(range(5, 100, 5))

# selecting loading test period
bicubic_ds = xr.open_dataset(bicubic_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))
bicubic = bicubic_ds[file_var].values

target_ds_var = xr.open_dataset(target_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))
target = target_ds_var[file_var].values

unet_pretrain = unet_pretrain_ds[var].sel(time=slice("2011-01-01", "2020-12-31")).values
unet_train = unet_train_ds[file_var].sel(time=slice("2011-01-01", "2020-12-31")).values

# Flattening for pooling plus thresholding
target_flat = target.flatten()
bicubic_flat = bicubic.flatten()
unet_pretrain_flat = unet_pretrain.flatten()
unet_train_flat = unet_train.flatten()

thresholds = [np.quantile(target_flat[~np.isnan(target_flat)], q/100) for q in quantiles]

def thresholded_mse(pred, target, thresholds, quantiles):
    mses = []
    for q, thresh in zip(quantiles, thresholds):
        mask = (target >= thresh)
        valid = mask & ~np.isnan(target) & ~np.isnan(pred)
        if np.sum(valid) == 0:
            mses.append(np.nan)
        else:
            mses.append(np.mean((pred[valid] - target[valid])**2))
    return mses

mse_bicubic = thresholded_mse(bicubic_flat, target_flat, thresholds, quantiles)
mse_pretrain = thresholded_mse(unet_pretrain_flat, target_flat, thresholds, quantiles)
mse_train = thresholded_mse(unet_train_flat, target_flat, thresholds, quantiles)

plt.figure(figsize=(10, 5))
plt.plot(quantiles, mse_bicubic, marker='o', color='blue', label="Bicubic")
plt.plot(quantiles, mse_pretrain, marker='o', color='red', label="UNet 1771 time series")
plt.plot(quantiles, mse_train, marker='o', color='green', label="UNet 1971 time series")
plt.xlabel("Quantile (%)")
plt.ylabel("Thresholded MSE")
plt.title(f"Thresholded MSE by Quantile for {var} ({file_var})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{config.OUTPUTS_DIR}/thresholded_mse_comparison_{var}.png", dpi=1000)
plt.close()

bicubic_ds.close()
target_ds_var.close()
unet_pretrain_ds.close()
unet_train_ds.close()