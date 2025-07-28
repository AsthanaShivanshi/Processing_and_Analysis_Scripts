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

fig, axes = plt.subplots(4, 3, figsize=(15, 18), constrained_layout=True)
all_stats = []

for row_idx, var in enumerate(var_list):
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

    # CVM and p each baseline
    for col_idx, model in enumerate([bicubic, unet_train, unet_combined]):
        stat, pval = gridwise_cvm_stat_p(model, target)
        all_stats.append(stat)
        ax = axes[row_idx, col_idx]
        # Rejected cells (p < 0.05)
        masked_stat = np.ma.masked_where(pval < 0.05, stat)
        im = ax.imshow(masked_stat, origin='lower', aspect='auto', cmap='viridis')
        # Overlaying rejected cells
        reject_mask = (pval < 0.05) & ~np.isnan(stat)
        if np.any(reject_mask):
            ax.imshow(np.where(reject_mask, 1, np.nan), cmap=mcolors.ListedColormap(['red']), 
                      origin='lower', aspect='auto', alpha=0.5, vmin=0, vmax=1)
        ax.set_title(f"{baseline_names[col_idx]}")
        ax.set_xticks([])
        ax.set_yticks([])
        if col_idx == 0:
            ax.set_ylabel(var.capitalize(), fontsize=14)

cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.015, pad=0.02)
cbar.set_label("Cramer–von Mises Test Statistic", fontsize=14)

fig.suptitle("Gridwise Cramer–von Mises Comparison\nRed: Rejected at 95% confidence", fontsize=18)
plt.savefig(f"{config.OUTPUTS_DIR}/Spatial/gridwise_cvm_comparison_grid.png", dpi=300)
plt.close()