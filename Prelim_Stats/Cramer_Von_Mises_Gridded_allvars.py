import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config
import matplotlib.colors as mcolors
from scipy.stats import cramervonmises_2samp

def gridwise_cvm_stat_p(a, b):
    stat = np.full(a.shape[1:], np.nan)
    pval = np.full(a.shape[1:], np.nan)
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            a1 = a[:, i, j]
            b1 = b[:, i, j]
            mask = ~np.isnan(a1) & ~np.isnan(b1)
            if np.sum(mask) > 1:
                try:
                    res = cramervonmises_2samp(a1[mask], b1[mask])
                    stat[i, j] = res.statistic
                    pval[i, j] = res.pvalue
                except Exception:
                    stat[i, j] = np.nan
                    pval[i, j] = np.nan
    return stat, pval

varnames = {
    "precip": "RhiresD",
    "temp": "TabsD",
    "tmin": "TminD",
    "tmax": "TmaxD"
}
var_list = list(varnames.keys())
baseline_names = ["Bicubic", "UNet 1971", "UNet Combined"]

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
    var: xr.open_dataset(bicubic_files[varnames[var]])
            [varnames[var]]
            .sel(time=slice("2011-01-01", "2020-12-31"))
            .values
    for var in var_list
}
unet_train = {
    var: xr.open_dataset(unet_train_path)
            [varnames[var]]
            .sel(time=slice("2011-01-01", "2020-12-31"))
            .values
    for var in var_list
}
unet_combined = {
    var: xr.open_dataset(unet_combined_path)
            [var] 
            .sel(time=slice("2011-01-01", "2020-12-31"))
            .values
    for var in var_list
}
target = {
    var: xr.open_dataset(target_files[varnames[var]])
            [varnames[var]]
            .sel(time=slice("2011-01-01", "2020-12-31"))
            .values
    for var in var_list
}

all_stats_flat = []
for row_idx, var in enumerate(var_list):
    for model in [bicubic[var], unet_train[var], unet_combined[var]]:
        stat, pval = gridwise_cvm_stat_p(model, target[var])
        all_stats_flat.append(stat[~np.isnan(stat)])
#vmin = np.nanmin(np.concatenate(all_stats_flat))
#vmax = np.nanmax(np.concatenate(all_stats_flat))

vmin = 0
vmax = 500

cividis = plt.colormaps['inferno']
colors = [cividis(i/256) for i in range(257)]
cmap = mcolors.ListedColormap(colors)
cmap.set_bad(color="white")  # NaN handling

# Load TabsD mask for Swiss domain
tabsd_ds = xr.open_dataset(target_files["TabsD"]).sel(time=slice("2011-01-01", "2020-12-31"))
tabsd_mask = ~np.isnan(tabsd_ds["TabsD"].isel(time=0).values)  # shape (lat, lon)

fig, axes = plt.subplots(4, 3, figsize=(15, 18), constrained_layout=True)

for row_idx, var in enumerate(var_list):
    for col_idx, model in enumerate([bicubic[var], unet_train[var], unet_combined[var]]):
        stat, pval = gridwise_cvm_stat_p(model, target[var])
        # Mask precipitation (top row, row_idx==0) to TabsD domain only
        if row_idx == 0:
            stat[~tabsd_mask] = np.nan
        ax = axes[row_idx, col_idx]
        im = ax.imshow(stat, origin='lower', aspect='auto', cmap=cmap)
        ax.set_title(f"{baseline_names[col_idx]}")
        ax.set_xticks([])
        ax.set_yticks([])
        if col_idx == 0:
            ax.set_ylabel(var.capitalize(), fontsize=14)

        # Compute and annotate pooled CvM statistic
        model_flat = model[~np.isnan(model) & ~np.isnan(target[var])]
        target_flat = target[var][~np.isnan(model) & ~np.isnan(target[var])]
        pooled_cvm = np.nan
        if len(model_flat) > 1 and len(target_flat) > 1:
            try:
                pooled_cvm = cramervonmises_2samp(model_flat, target_flat).statistic
            except Exception:
                pooled_cvm = np.nan
        ax.text(
            0.02, 0.98,
            f"Pooled CvM: {pooled_cvm:.3f}" if not np.isnan(pooled_cvm) else "Pooled CvM: nan",
            color="black", fontsize=11, fontweight="bold",
            ha="left", va="top", transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
        )
cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.015, pad=0.02)
cbar.set_label("Cramer–von Mises Test Statistic", fontsize=14)

print(f"CvM statistic range: {vmin:.4f} to {vmax:.4f}")

fig.suptitle("Gridwise Cramer–von Mises Statistic on test set (2011-2020)", fontsize=24, fontweight='bold')
plt.savefig(f"{config.OUTPUTS_DIR}/Spatial/gridwise_cvm.png", dpi=1000)
plt.close()