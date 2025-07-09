import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config
import argparse
import os

parser = argparse.ArgumentParser(description="Thresholded MSE Spatial Maps and Quantile MSE Curves")
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
unet_combined_path= f"{config.UNET_COMBINED_DIR}/Combined_Dataset_Downscaled_Predictions_2011_2020.nc"
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
unet_combined_ds = xr.open_dataset(unet_combined_path)

bicubic_ds = xr.open_dataset(bicubic_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))
bicubic = bicubic_ds[file_var].values

target_ds_var = xr.open_dataset(target_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))
target = target_ds_var[file_var].values

unet_pretrain = unet_pretrain_ds[var].sel(time=slice("2011-01-01", "2020-12-31")).values
unet_train = unet_train_ds[file_var].sel(time=slice("2011-01-01", "2020-12-31")).values
unet_combined = unet_combined_ds[var].sel(time=slice("2011-01-01", "2020-12-31")).values

valid_mask = ~np.isnan(target) & ~np.isnan(bicubic) & ~np.isnan(unet_pretrain) & ~np.isnan(unet_train)
target = np.where(valid_mask, target, np.nan)
bicubic = np.where(valid_mask, bicubic, np.nan)
unet_pretrain = np.where(valid_mask, unet_pretrain, np.nan)
unet_train = np.where(valid_mask, unet_train, np.nan)
unet_combined = np.where(valid_mask, unet_combined, np.nan)

# Spatial MSE maps at selected quantiles---
quantiles_to_plot = [5, 25, 50, 75, 95, 99]
thresholds = [np.nanquantile(target, q/100) for q in quantiles_to_plot]

mse_maps = {
    "Bicubic": [],
    "UNet 1771": [],
    "UNet 1971": [],
    "UNet Combined": []
}

for thresh in thresholds:
    mask = (target <= thresh)
    def mse_map(pred):
        squared_error = (pred - target) ** 2
        squared_error_masked = np.where(mask, squared_error, np.nan)
        return np.nanmean(squared_error_masked, axis=0)
    mse_maps["Bicubic"].append(mse_map(bicubic))
    mse_maps["UNet 1771"].append(mse_map(unet_pretrain))
    mse_maps["UNet 1971"].append(mse_map(unet_train))
    mse_maps["UNet Combined"].append(mse_map(unet_combined))

all_maps = np.array(
    [mse_maps[m][i] for i in range(len(quantiles_to_plot)) for m in ["Bicubic", "UNet 1771", "UNet 1971", "UNet Combined"]]
)
vmin = np.nanmin(all_maps)
vmax = np.nanmax(all_maps)

fig, axes = plt.subplots(4, 3, figsize=(15, 18), constrained_layout=True)
method_names = ["Bicubic", "UNet 1771", "UNet 1971", "UNet Combined"]

for i, q in enumerate(quantiles_to_plot):
    for j, method in enumerate(method_names):
        ax = axes[i, j]
        im = ax.imshow(mse_maps[method][i], origin='lower', aspect='auto', cmap='RdYlBu', vmin=vmin, vmax=vmax)
        ax.set_title(f"{method}\n{q}th percentile")
        ax.set_xticks([])
        ax.set_yticks([])

cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.025, pad=0.02)
cbar.set_label("MSE")

fig.suptitle(f"Spatial MSE Maps at Selected Quantiles for {var} ({file_var})", fontsize=18)
plt.savefig(f"{config.OUTPUTS_DIR}/spatial_mse_maps_{var}.png", dpi=300)
plt.close()