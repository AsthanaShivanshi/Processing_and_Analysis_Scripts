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

    # RMSE for each grid cell over the entire time series
    bicubic_rmse = np.sqrt(np.nanmean((bicubic - target) ** 2, axis=0))
    unet_train_rmse = np.sqrt(np.nanmean((unet_train - target) ** 2, axis=0))
    unet_combined_rmse = np.sqrt(np.nanmean((unet_combined - target) ** 2, axis=0))

    diff_1971 = bicubic_rmse - unet_train_rmse
    diff_combined = bicubic_rmse - unet_combined_rmse
    green_mask = diff_1971 > diff_combined  # 1971 is better
    blue_mask = ~green_mask                  # Combined is better
    color_map = np.full(diff_1971.shape, np.nan, dtype=object)
    color_map[green_mask] = "green"
    color_map[blue_mask] = "blue"

    ax = axes[idx]
    plot_map = np.full(diff_1971.shape, np.nan)
    plot_map[green_mask] = 1
    plot_map[blue_mask] = 0
    cmap = mcolors.ListedColormap(["blue", "green"])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(plot_map, origin='lower', aspect='auto', cmap=cmap, norm=norm)
    ax.set_title(f"{var.capitalize()} - Green: 1971 better, Blue: Combined better")
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("Gridwise RMSE Comparison: 1971 vs Combined (Over bicubic baseline)", fontsize=18)
plt.savefig(f"{config.OUTPUTS_DIR}/Spatial/gridwise_rmse_comparison.png", dpi=1000)
plt.close()