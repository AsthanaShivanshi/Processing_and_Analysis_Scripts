import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config
import argparse
import os
from matplotlib.ticker import MaxNLocator, FuncFormatter

parser = argparse.ArgumentParser(description="Spatial Quantile Bias Spatial Maps")
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

unet_train_path = f"{config.UNET_1971_DIR}/Optim_Training_Downscaled_Predictions_2011_2020.nc"
unet_combined_path = f"{config.UNET_COMBINED_DIR}/Combined_Downscaled_Predictions_2011_2020.nc"
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

unet_train_ds = xr.open_dataset(unet_train_path)
unet_combined_ds = xr.open_dataset(unet_combined_path)
bicubic_ds = xr.open_dataset(bicubic_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))
target_ds_var = xr.open_dataset(target_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))

target = target_ds_var[file_var].values
bicubic = bicubic_ds[file_var].values
unet_train = unet_train_ds[file_var].sel(time=slice("2011-01-01", "2020-12-31")).values
unet_combined = unet_combined_ds[var].sel(time=slice("2011-01-01", "2020-12-31")).values

valid_mask = ~np.isnan(target) & ~np.isnan(bicubic) & ~np.isnan(unet_train)
target = np.where(valid_mask, target, np.nan)
bicubic = np.where(valid_mask, bicubic, np.nan)
unet_train = np.where(valid_mask, unet_train, np.nan)
unet_combined = np.where(valid_mask, unet_combined, np.nan)

quantiles_to_plot = [5, 50, 95, 99]
qvals = [q/100 for q in quantiles_to_plot]

bias_maps = {
    "Bicubic": [],
    "UNet 1971": [],
    "UNet Combined": []
}

for q in qvals:
    if var == "precip":
        # Remove zeros for quantile calculation only
        target_masked = np.where(target == 0, np.nan, target)
        bicubic_masked = np.where(bicubic == 0, np.nan, bicubic)
        unet_train_masked = np.where(unet_train == 0, np.nan, unet_train)
        unet_combined_masked = np.where(unet_combined == 0, np.nan, unet_combined)
    else:
        target_masked = target
        bicubic_masked = bicubic
        unet_train_masked = unet_train
        unet_combined_masked = unet_combined

    # Compute quantile over time axis (axis=0)
    target_q = np.nanquantile(target_masked, q, axis=0)
    bicubic_q = np.nanquantile(bicubic_masked, q, axis=0)
    unet_train_q = np.nanquantile(unet_train_masked, q, axis=0)
    unet_combined_q = np.nanquantile(unet_combined_masked, q, axis=0)

    bias_maps["Bicubic"].append(bicubic_q - target_q)
    bias_maps["UNet 1971"].append(unet_train_q - target_q)
    bias_maps["UNet Combined"].append(unet_combined_q - target_q)

# Plotting the spatial quantile bias maps wrt observations
method_names = ["Bicubic", "UNet 1971", "UNet Combined"]
nrows = len(quantiles_to_plot)
ncols = len(method_names)

all_maps = np.array(
    [bias_maps[m][i] for m in method_names for i in range(nrows)]
)
if var == "precip":
    vmin, vmax = -20, 20
    cmap = "BrBG"
    title_label= "Non Zero Precipitation Quantile Bias"
else:
    vmin = -10
    vmax = 10
    cmap = "coolwarm"
    title_label = f"{var.capitalize()} Quantile Bias"


fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), constrained_layout=True)

for j, method in enumerate(method_names):
    # Set colorbar range and colormap per method for temperature variables
    if var != "precip":
        if method == "Bicubic":
            vmin, vmax, cmap, tick_step = -10, 10, "coolwarm", 1
        elif method == "UNet 1971":
            vmin, vmax, cmap, tick_step = -1, 1, "coolwarm", 0.2
        elif method == "UNet Combined":
            vmin, vmax, cmap, tick_step = -10, 10, "coolwarm", 1
    for i, q in enumerate(quantiles_to_plot):
        ax = axes[i, j]
        im = ax.imshow(bias_maps[method][i], origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        if i == 0:
            ax.set_title(method)
        if j == 0:
            ax.set_ylabel(f"{q}th percentile")
        ax.set_xticks([])
        ax.set_yticks([])


    if i == nrows - 1:
        cbar = fig.colorbar(im, ax=[axes[k, j] for k in range(nrows)], orientation='vertical', fraction=0.025, pad=0.02)
        cbar.set_label("Quantile Bias (Model - Obs)")
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))

fig.suptitle(f"Spatial Quantile Bias Maps for {var} ({file_var})", fontsize=16)
plt.savefig(f"{config.OUTPUTS_DIR}/Spatial/spatial_quantile_bias_maps_{var}.png", dpi=1500)
plt.close()