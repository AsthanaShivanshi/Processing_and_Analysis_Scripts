import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config
import argparse
from matplotlib.ticker import MaxNLocator, FuncFormatter

parser = argparse.ArgumentParser(description="Spatial Threshold RMSE Maps")
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

valid_mask = ~np.isnan(target) & ~np.isnan(bicubic) & ~np.isnan(unet_train) & ~np.isnan(unet_combined)
target = np.where(valid_mask, target, np.nan)
bicubic = np.where(valid_mask, bicubic, np.nan)
unet_train = np.where(valid_mask, unet_train, np.nan)
unet_combined = np.where(valid_mask, unet_combined, np.nan)

quantiles_to_plot = [5, 50, 95, 99]
qvals = [q/100 for q in quantiles_to_plot]

rmse_maps = {
    "Bicubic": [],
    "UNet 1971": [],
    "UNet Combined": []
}

winner_maps = []

for var in var_list:
    file_var = varnames[var]
    bicubic = bicubic_ds[file_var].values
    unet_train = unet_train_ds[file_var].sel(time=slice("2011-01-01", "2020-12-31")).values
    target = xr.open_dataset(target_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))[file_var].values
    unet_combined = unet_combined_ds[var].sel(time=slice("2011-01-01", "2020-12-31")).values

    valid_mask = ~np.isnan(target) & ~np.isnan(bicubic) & ~np.isnan(unet_train) & ~np.isnan(unet_combined)
    target = np.where(valid_mask, target, np.nan)
    bicubic = np.where(valid_mask, bicubic, np.nan)
    unet_train = np.where(valid_mask, unet_train, np.nan)
    unet_combined = np.where(valid_mask, unet_combined, np.nan)

    var_winner_maps = []
    for q in qvals:
        target_q = np.nanquantile(target, q, axis=0)

        bicubic_rmse = np.sqrt(np.nanmean((np.where(target <= target_q, bicubic - target, np.nan))**2, axis=0))
        unet_train_rmse = np.sqrt(np.nanmean((np.where(target <= target_q, unet_train - target, np.nan))**2, axis=0))
        unet_combined_rmse = np.sqrt(np.nanmean((np.where(target <= target_q, unet_combined - target, np.nan))**2, axis=0))

        winner = np.full(target_q.shape, np.nan)
        winner[(unet_train_rmse < bicubic_rmse) & (unet_train_rmse < unet_combined_rmse)] = 0
        winner[(unet_combined_rmse < bicubic_rmse) & (unet_combined_rmse < unet_train_rmse)] = 1
        winner[~((unet_train_rmse < bicubic_rmse) & (unet_train_rmse < unet_combined_rmse)) &
               ~((unet_combined_rmse < bicubic_rmse) & (unet_combined_rmse < unet_train_rmse))] = 2
        var_winner_maps.append(winner)
    winner_maps.append(var_winner_maps)

winner_maps = np.array(winner_maps)  # shape (n_vars, n_percentiles, lat, lon)

nrows = len(var_list)
ncols = len(quantiles_to_plot)
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), constrained_layout=True)

cmap = plt.matplotlib.colors.ListedColormap(["#003366", "#FF7F50", "#F0F0F0"])  
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

for i in range(nrows):
    for j in range(ncols):
        ax = axes[i, j]
        im = ax.imshow(winner_maps[i, j], origin='lower', aspect='auto', cmap=cmap, norm=norm)
        if i == 0:
            ax.set_title(f"{quantiles_to_plot[j]}th percentile")
        if j == 0:
            ax.set_ylabel(var_list[i].capitalize())
        ax.set_xticks([])
        ax.set_yticks([])

cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.025, pad=0.02, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(["UNet 1971 better", "UNet Combined better", "Neither over bicubic"])
cbar.set_label("Gridwise Winner (Thresholded RMSE)", fontsize=14)

fig.suptitle("Gridwise Model Comparison: Thresholded RMSE (UNet 1971 vs Combined vs Bicubic)", fontsize=24, weight='bold')
plt.savefig(f"{config.OUTPUTS_DIR}/Spatial/spatial_thresholded_rmse_comparison.png", dpi=1000)
plt.close()