import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config
import argparse
from skimage import measure
from scipy.ndimage import map_coordinates

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

quantiles_to_plot = [5, 50, 95, 99]
qvals = [q/100 for q in quantiles_to_plot]

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

tabsd_ds = xr.open_dataset(target_files["TabsD"]).sel(time=slice("2011-01-01", "2020-12-31"))
tabsd_mask = ~np.isnan(tabsd_ds["TabsD"].isel(time=0).values) 
contours = measure.find_contours(tabsd_mask.astype(float), 0.5)
lats = tabsd_ds["lat"].values
lons = tabsd_ds["lon"].values
lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

winner_maps = []
for var in var_list:
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

    var_winner_maps = []
    for q in qvals:
        if var == "precip":
            target_masked = np.where(target == 0, np.nan, target)
            unet_train_masked = np.where(unet_train == 0, np.nan, unet_train)
            unet_combined_masked = np.where(unet_combined == 0, np.nan, unet_combined)
        else:
            target_masked = target
            unet_train_masked = unet_train
            unet_combined_masked = unet_combined

        target_q = np.nanquantile(target_masked, q, axis=0)
        unet_train_q = np.nanquantile(unet_train_masked, q, axis=0)
        unet_combined_q = np.nanquantile(unet_combined_masked, q, axis=0)

        abs_bias_1971 = np.abs(unet_train_q - target_q)
        abs_bias_combined = np.abs(unet_combined_q - target_q)

        winner = np.full(target_q.shape, np.nan)
        winner[abs_bias_1971 < abs_bias_combined] = 0  # UNet 1971 better
        winner[abs_bias_combined < abs_bias_1971] = 1  # UNet Combined better
        winner[abs_bias_1971 == abs_bias_combined] = 2  # Tied

        var_winner_maps.append(winner)
    winner_maps.append(var_winner_maps)

winner_maps = np.array(winner_maps)

nrows = len(var_list)
ncols = len(quantiles_to_plot)
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), constrained_layout=True)

cmap = plt.matplotlib.colors.ListedColormap(["#1f77b4", "#ff7f0e", "#FFFFFF"])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

for i in range(nrows):
    for j in range(ncols):
        ax = axes[i, j]
        im = ax.imshow(winner_maps[i, j], origin='lower', aspect='auto', cmap=cmap, norm=norm)
        if i == 0:
            # Overlay Switzerland boundary using interpolated coordinates
            for contour in contours:
                contour_lat = map_coordinates(lat_grid, [contour[:, 0], contour[:, 1]], order=1)
                contour_lon = map_coordinates(lon_grid, [contour[:, 0], contour[:, 1]], order=1)
                ax.plot(contour_lon, contour_lat, color='red', linestyle=':', linewidth=2, zorder=10)
            ax.set_title(f"{quantiles_to_plot[j]}th percentile")
        if j == 0:
            ax.set_ylabel(var_list[i].capitalize())
        ax.set_xticks([])
        ax.set_yticks([])

cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.025, pad=0.02, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(["UNet 1971 better", "UNet Combined better", "Neither over Bicubic"])
cbar.set_label("Model with lower Quantile Bias", fontsize=14)

fig.suptitle("Gridwise Model Comparison: Quantile Bias (UNet 1971 vs Combined)", fontsize=24, weight='bold')
plt.savefig(f"{config.OUTPUTS_DIR}/Spatial/spatial_quantile_bias_comparison.png", dpi=1000)