import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config
import matplotlib.colors as mcolors

varnames = {
    "precip": "RhiresD",
    "temp": "TabsD",
    "tmin": "TminD",
    "tmax": "TmaxD"
}
var_list = list(varnames.keys())

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

fig, axes = plt.subplots(4, 1, figsize=(10, 16), constrained_layout=True)
for idx, var in enumerate(var_list):
    file_var = varnames[var]
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

    bicubic_rmse = np.sqrt(np.nanmean((bicubic - target) ** 2, axis=0))
    unet_train_rmse = np.sqrt(np.nanmean((unet_train - target) ** 2, axis=0))
    unet_combined_rmse = np.sqrt(np.nanmean((unet_combined - target) ** 2, axis=0))

    # Tripartite logic
    plot_map = np.full(bicubic_rmse.shape, np.nan)
    # 0: UNet 1971 better, 1: UNet Combined, 2: Neither better (bicubic best or tied)
    is_1971_better = (unet_train_rmse < bicubic_rmse) & (unet_train_rmse < unet_combined_rmse)
    is_combined_better = (unet_combined_rmse < bicubic_rmse) & (unet_combined_rmse < unet_train_rmse)
    is_neither_better = ~(is_1971_better | is_combined_better)
    plot_map[is_1971_better] = 0
    plot_map[is_combined_better] = 1
    plot_map[is_neither_better] = 2

    cmap = mcolors.ListedColormap(["#009E73", "#CC79A7", "#FFFFFF"])  # Green, Purple, White
    cmap.set_bad(color="white")
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    im = ax = axes[idx].imshow(plot_map, origin='lower', aspect='auto', cmap=cmap, norm=norm)
    axes[idx].set_title(f"{var.capitalize()} - Green: 1971 better, Purple: Combined better, White: Bicubic best/tied")
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])

cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.015, pad=0.02, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(["UNet 1971 better", "UNet Combined better", "Bicubic best/tied"])
cbar.set_label("Model with lowest RMSE", fontsize=14)

fig.suptitle("Gridwise RMSE Comparison: UNet 1971 vs Combined vs Bicubic", fontsize=18)
plt.savefig(f"{config.OUTPUTS_DIR}/Spatial/gridwise_rmse_comparison.png", dpi=1000)
plt.close()