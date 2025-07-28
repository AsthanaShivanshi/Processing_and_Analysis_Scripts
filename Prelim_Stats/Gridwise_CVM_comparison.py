import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config
import matplotlib.colors as mcolors
from scipy.stats import cramervonmises_2samp

def gridwise_cvm_stat(a, b):
    stat = np.full(a.shape[1:], np.nan)
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            a1 = a[:, i, j]
            b1 = b[:, i, j]
            mask = ~np.isnan(a1) & ~np.isnan(b1)
            if np.sum(mask) > 1:
                try:
                    stat[i, j] = cramervonmises_2samp(a1[mask], b1[mask]).statistic
                except Exception:
                    stat[i, j] = np.nan
    return stat

varnames = {
    "precip": "RhiresD",
    "temp": "TabsD",
    "tmin": "TminD",
    "tmax": "TmaxD"
}
var_list = list(varnames.keys())
baseline_names = ["UNet 1971 better", "UNet Combined better"]

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

bicubic = {
    var: xr.open_dataset(bicubic_files[varnames[var]])[varnames[var]].sel(time=slice("2011-01-01", "2020-12-31")).values
    for var in var_list
}
unet_train = {
    var: xr.open_dataset(unet_train_path)[varnames[var]].sel(time=slice("2011-01-01", "2020-12-31")).values
    for var in var_list
}
unet_combined = {
    var: xr.open_dataset(unet_combined_path)[var].sel(time=slice("2011-01-01", "2020-12-31")).values
    for var in var_list
}
target = {
    var: xr.open_dataset(target_files[varnames[var]])[varnames[var]].sel(time=slice("2011-01-01", "2020-12-31")).values
    for var in var_list
}

winner_maps = []

for var in var_list:
    # Compute CvM stats for each model vs target
    stat_bicubic = gridwise_cvm_stat(bicubic[var], target[var])
    stat_unet_train = gridwise_cvm_stat(unet_train[var], target[var])
    stat_unet_combined = gridwise_cvm_stat(unet_combined[var], target[var])

    # For each grid cell: 0 if UNet 1971 is closer to target than bicubic, 1 if UNet Combined is closer, np.nan if neither
    winner = np.full(stat_bicubic.shape, np.nan)
    better_1971 = (stat_unet_train < stat_bicubic) & (stat_unet_train < stat_unet_combined)
    better_combined = (stat_unet_combined < stat_bicubic) & (stat_unet_combined < stat_unet_train)
    winner[better_1971] = 0
    winner[better_combined] = 1
    winner_maps.append(winner)

fig, axes = plt.subplots(4, 1, figsize=(8, 16), constrained_layout=True)
cmap = mcolors.ListedColormap(["#0072B2", "#E69F00"])  # Blue, Orange
bounds = [-0.5, 0.5, 1.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
for idx, (ax, winner) in enumerate(zip(axes, winner_maps)):
    im = ax.imshow(winner, origin='lower', aspect='auto', cmap=cmap, norm=norm)
    ax.set_title(var_list[idx].capitalize())
    ax.set_xticks([])
    ax.set_yticks([])

cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.015, pad=0.02, ticks=[0, 1])
cbar.ax.set_yticklabels(["UNet 1971 better", "UNet Combined better"])
cbar.set_label("Model with lower CvM statistic than Bicubic", fontsize=14)

fig.suptitle("Gridwise: Which UNet is closer to target than Bicubic (Cramér–von Mises)", fontsize=16)
plt.savefig(f"{config.OUTPUTS_DIR}/Spatial/gridwise_cvm_comparison.png", dpi=1000)
plt.close()