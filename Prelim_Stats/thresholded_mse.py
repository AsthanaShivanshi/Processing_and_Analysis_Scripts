import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config
import argparse
import os

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

bicubic_ds = xr.open_dataset(bicubic_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))
bicubic = bicubic_ds[file_var].values

target_ds_var = xr.open_dataset(target_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))
target = target_ds_var[file_var].values

unet_pretrain = unet_pretrain_ds[var].sel(time=slice("2011-01-01", "2020-12-31")).values
unet_train = unet_train_ds[file_var].sel(time=slice("2011-01-01", "2020-12-31")).values

# Check shapes and NaN counts before flattening
print("target shape:", target.shape, "NaNs:", np.isnan(target).sum())
print("bicubic shape:", bicubic.shape, "NaNs:", np.isnan(bicubic).sum())
print("unet_pretrain shape:", unet_pretrain.shape, "NaNs:", np.isnan(unet_pretrain).sum())
print("unet_train shape:", unet_train.shape, "NaNs:", np.isnan(unet_train).sum())

# Check if NaN masks are aligned
print("NaN mask equal (bicubic vs target):", np.array_equal(np.isnan(bicubic), np.isnan(target)))
print("NaN mask equal (unet_pretrain vs target):", np.array_equal(np.isnan(unet_pretrain), np.isnan(target)))
print("NaN mask equal (unet_train vs target):", np.array_equal(np.isnan(unet_train), np.isnan(target)))

# Check for all-NaN time steps and grid points
if target.ndim == 3:
    print("Any all-NaN time steps in target:", np.any(np.all(np.isnan(target), axis=(1,2))))
    print("Any all-NaN grid points in target:", np.any(np.all(np.isnan(target), axis=0)))

# Flatten for thresholding
target_flat = target.flatten()
bicubic_flat = bicubic.flatten()
unet_pretrain_flat = unet_pretrain.flatten()
unet_train_flat = unet_train.flatten()

# Check for all-NaN after flattening
print("target_flat NaNs:", np.isnan(target_flat).sum(), "/", target_flat.size)
print("bicubic_flat NaNs:", np.isnan(bicubic_flat).sum(), "/", bicubic_flat.size)
print("unet_pretrain_flat NaNs:", np.isnan(unet_pretrain_flat).sum(), "/", unet_pretrain_flat.size)
print("unet_train_flat NaNs:", np.isnan(unet_train_flat).sum(), "/", unet_train_flat.size)

print("target_flat min/max:", np.nanmin(target_flat), np.nanmax(target_flat))
print("bicubic_flat min/max:", np.nanmin(bicubic_flat), np.nanmax(bicubic_flat))
print("unet_pretrain_flat min/max:", np.nanmin(unet_pretrain_flat), np.nanmax(unet_pretrain_flat))
print("unet_train_flat min/max:", np.nanmin(unet_train_flat), np.nanmax(unet_train_flat))

print("Sample (bicubic - target):", (bicubic_flat - target_flat)[:10])
print("Sample (unet_pretrain - target):", (unet_pretrain_flat - target_flat)[:10])
print("Sample (unet_train - target):", (unet_train_flat - target_flat)[:10])

thresholds = [np.quantile(target_flat[~np.isnan(target_flat)], q/100) for q in quantiles]
print("Thresholds:", thresholds)

def thresholded_mse(pred, target, thresholds, quantiles, label=""):
    mses = []
    for q, thresh in zip(quantiles, thresholds):
        mask = (target >= thresh)
        valid = mask & ~np.isnan(target) & ~np.isnan(pred)
        print(f"{label} Quantile {q}: threshold={thresh}, valid points={np.sum(valid)}")
        if np.sum(valid) == 0:
            mses.append(np.nan)
        else:
            mses.append(np.mean((pred[valid] - target[valid])**2))
    return mses

mse_bicubic = thresholded_mse(bicubic_flat, target_flat, thresholds, quantiles, label="Bicubic")
mse_pretrain = thresholded_mse(unet_pretrain_flat, target_flat, thresholds, quantiles, label="UNet 1771")
mse_train = thresholded_mse(unet_train_flat, target_flat, thresholds, quantiles, label="UNet 1971")

diffs = np.abs(bicubic_flat - target_flat)
print("Top 10 largest abs(bicubic - target):", np.sort(diffs)[-10:])

print("First 10 target values:", target_flat[:10])
print("First 10 bicubic values:", bicubic_flat[:10])

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