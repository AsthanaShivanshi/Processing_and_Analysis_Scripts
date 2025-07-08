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

bicubic_ds = xr.open_dataset(bicubic_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))
bicubic = bicubic_ds[file_var].values

target_ds_var = xr.open_dataset(target_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))
target = target_ds_var[file_var].values

unet_pretrain = unet_pretrain_ds[var].sel(time=slice("2011-01-01", "2020-12-31")).values
unet_train = unet_train_ds[file_var].sel(time=slice("2011-01-01", "2020-12-31")).values

valid_mask = ~np.isnan(target) & ~np.isnan(bicubic) & ~np.isnan(unet_pretrain) & ~np.isnan(unet_train)
target = np.where(valid_mask, target, np.nan)
bicubic = np.where(valid_mask, bicubic, np.nan)
unet_pretrain = np.where(valid_mask, unet_pretrain, np.nan)
unet_train = np.where(valid_mask, unet_train, np.nan)

# --- Quantile MSE curves (NEW LOGIC) ---
quantiles = list(range(5, 100, 5))
method_arrays = {
    "Bicubic": bicubic,
    "UNet 1771": unet_pretrain,
    "UNet 1971": unet_train
}

for method, pred in method_arrays.items():
    preds_flat = pred.flatten()
    targets_flat = target.flatten()
    mses = []
    for q in quantiles:
        threshold = np.nanquantile(targets_flat, q / 100)
        mask = targets_flat <= threshold
        if np.sum(mask) == 0:
            mses.append(np.nan)
        else:
            mses.append(np.nanmean((preds_flat[mask] - targets_flat[mask]) ** 2))
    plt.plot(quantiles, mses, marker='o', label=method)

plt.xlabel("Quantile (%)")
plt.ylabel("MSE")
plt.title(f"MSE below quantile for {var} ({file_var})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{config.OUTPUTS_DIR}/{file_var}_quantile_mse.png", dpi=300)
plt.close()

# --- Spatial MSE maps at selected quantiles (original logic) ---
quantiles_to_plot = [5, 50, 95, 99]
thresholds = [np.nanquantile(target, q/100) for q in quantiles_to_plot]

mse_maps = {
    "Bicubic": [],
    "UNet 1771": [],
    "UNet 1971": []
}

for thresh in thresholds:
    mask = (target >= thresh)
    def mse_map(pred):
        squared_error = (pred - target) ** 2
        squared_error_masked = np.where(mask, squared_error, np.nan)
        return np.nanmean(squared_error_masked, axis=0)
    mse_maps["Bicubic"].append(mse_map(bicubic))
    mse_maps["UNet 1771"].append(mse_map(unet_pretrain))
    mse_maps["UNet 1971"].append(mse_map(unet_train))

all_maps = np.array(
    [mse_maps[m][i] for i in range(len(quantiles_to_plot)) for m in ["Bicubic", "UNet 1771", "UNet 1971"]]
)
vmin = np.nanmin(all_maps)
vmax = np.nanmax(all_maps)

fig, axes = plt.subplots(4, 3, figsize=(15, 18), constrained_layout=True)
method_names = ["Bicubic", "UNet 1771", "UNet 1971"]

for i, q in enumerate(quantiles_to_plot):
    for j, method in enumerate(method_names):
        ax = axes[i, j]
        im = ax.imshow(mse_maps[method][i], origin='lower', aspect='auto', cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax.set_title(f"{method}\n{q}th percentile")
        ax.set_xticks([])
        ax.set_yticks([])

cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.025, pad=0.02)
cbar.set_label("MSE")

fig.suptitle(f"Spatial MSE Maps at Selected Quantiles for {var} ({file_var})", fontsize=18)
plt.savefig(f"{config.OUTPUTS_DIR}/spatial_mse_maps_{var}.png", dpi=300)
plt.close()