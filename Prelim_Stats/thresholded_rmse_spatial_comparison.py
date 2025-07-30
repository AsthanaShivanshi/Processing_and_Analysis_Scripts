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

quantiles_to_plot = [5, 50, 95, 99]
qvals = [q/100 for q in quantiles_to_plot]

rmse_maps = {
    "Bicubic": [],
    "UNet 1971": [],
    "UNet Combined": []
}

plot_labels = ["Precip", "Temp", "Tmin", "Tmax"]
winner_maps = []

for i, var in enumerate(var_list):
    file_var = varnames[var]
    bicubic_ds = xr.open_dataset(bicubic_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))
    target_ds_var = xr.open_dataset(target_files[file_var]).sel(time=slice("2011-01-01", "2020-12-31"))

    bicubic = bicubic_ds[file_var].values
    unet_train = unet_train_ds[file_var].sel(time=slice("2011-01-01", "2020-12-31")).values
    target = target_ds_var[file_var].values
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
        valid_grid = ~np.isnan(bicubic_rmse) & ~np.isnan(unet_train_rmse) & ~np.isnan(unet_combined_rmse)
        winner[(unet_train_rmse < bicubic_rmse) & (unet_train_rmse < unet_combined_rmse) & valid_grid] = 0
        winner[(unet_combined_rmse < bicubic_rmse) & (unet_combined_rmse < unet_train_rmse) & valid_grid] = 1
        winner[~((unet_train_rmse < bicubic_rmse) & (unet_train_rmse < unet_combined_rmse)) &
               ~((unet_combined_rmse < bicubic_rmse) & (unet_combined_rmse < unet_train_rmse)) & valid_grid] = 2
        var_winner_maps.append(winner)
    winner_maps.append(var_winner_maps)

winner_maps = np.array(winner_maps)

cmap = plt.matplotlib.colors.ListedColormap(["#003366", "#FF7F50", "#A9A9A9"])
cmap.set_bad(color="#FFFFFF")
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

fig, axes = plt.subplots(len(var_list), len(quantiles_to_plot), figsize=(4*len(quantiles_to_plot), 3*len(var_list)), constrained_layout=True)
for i in range(len(var_list)):
    for j in range(len(quantiles_to_plot)):
        ax = axes[i, j]
        data = winner_maps[i, j]
        # Masking invalids before plotting : white v grey
        masked_data = np.ma.masked_invalid(data)
        im = ax.imshow(masked_data, origin='lower', aspect='auto', cmap=cmap, norm=norm)
        if i == 0:
            ax.set_title(f"{quantiles_to_plot[j]}th percentile")
        if j == 0:
            ax.set_ylabel(plot_labels[i])
        ax.set_xticks([])
        ax.set_yticks([])

cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes[1:, :], orientation='vertical', fraction=0.025, pad=0.02, ticks=[0, 1, 2]
)
cbar.ax.set_yticklabels(["UNet 1971 better", "UNet Combined better", "Neither over bicubic"])

fig.suptitle("Gridwise Model Comparison: Thresholded RMSE (UNet 1971 vs Combined vs Bicubic)", fontsize=24, weight='bold')
plt.savefig(f"{config.OUTPUTS_DIR}/Spatial/spatial_thresholded_rmse_comparison.png", dpi=1000)
plt.close()