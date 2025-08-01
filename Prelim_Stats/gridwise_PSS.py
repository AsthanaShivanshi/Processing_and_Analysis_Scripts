#Gridwise Perkins SS
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config
import matplotlib.colors as mcolors

def gridwise_perkins_skill_score(a, b, nbins=50):
    pss = np.full(a.shape[1:], np.nan)
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            a1 = a[:, i, j]
            b1 = b[:, i, j]
            mask = ~np.isnan(a1) & ~np.isnan(b1)
            if np.sum(mask) > 10:
                try:
                    a_valid = a1[mask]
                    b_valid = b1[mask]
                    combined_data = np.concatenate([a_valid, b_valid])
                    bins = np.linspace(np.min(combined_data), np.max(combined_data), nbins + 1)
                    hist_a, _ = np.histogram(a_valid, bins=bins, density=True)
                    hist_b, _ = np.histogram(b_valid, bins=bins, density=True)
                    hist_a = hist_a / np.sum(hist_a)
                    hist_b = hist_b / np.sum(hist_b)
                    pss[i, j] = np.sum(np.minimum(hist_a, hist_b))
                except Exception:
                    pss[i, j] = np.nan
    return pss

def pooled_perkins_skill_score(a, b, nbins=50):
    mask = ~np.isnan(a) & ~np.isnan(b)
    a_flat = a[mask]
    b_flat = b[mask]
    if len(a_flat) > 10 and len(b_flat) > 10:
        try:
            combined_data = np.concatenate([a_flat, b_flat])
            bins = np.linspace(np.min(combined_data), np.max(combined_data), nbins + 1)
            hist_a, _ = np.histogram(a_flat, bins=bins, density=True)
            hist_b, _ = np.histogram(b_flat, bins=bins, density=True)
            hist_a = hist_a / np.sum(hist_a)
            hist_b = hist_b / np.sum(hist_b)
            return np.sum(np.minimum(hist_a, hist_b))
        except Exception:
            return np.nan
    else:
        return np.nan

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

# TabsD mask for Swiss domain
tabsd_ds = xr.open_dataset(target_files["TabsD"]).sel(time=slice("2011-01-01", "2020-12-31"))
tabsd_mask = ~np.isnan(tabsd_ds["TabsD"].isel(time=0).values)  # shape (lat, lon)

vmin = 0
vmax = 1

coolwarm = plt.colormaps['PiYG']
cmap = mcolors.ListedColormap(coolwarm(np.linspace(0, 1, 256)))
cmap.set_bad(color="white")  # NaN handling

fig, axes = plt.subplots(4, 3, figsize=(15, 18), constrained_layout=True)

for row_idx, var in enumerate(var_list):
    for col_idx, model in enumerate([bicubic[var], unet_train[var], unet_combined[var]]):
        pss = gridwise_perkins_skill_score(model, target[var])
        # Mask precipitation (top row, row_idx==0) to TabsD domain only
        if row_idx == 0:
            pss[~tabsd_mask] = np.nan
        ax = axes[row_idx, col_idx]
        im = ax.imshow(pss, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{baseline_names[col_idx]}")
        ax.set_xticks([])
        ax.set_yticks([])
        if col_idx == 0:
            ax.set_ylabel(var.capitalize(), fontsize=14)

        # Annotate pooled PSS in top left
        pooled_pss = pooled_perkins_skill_score(model, target[var])
        ax.text(
            0.02, 0.98,
            f"Pooled PSS: {pooled_pss:.3f}" if not np.isnan(pooled_pss) else "Pooled PSS: nan",
            color="black", fontsize=11, fontweight="bold",
            ha="left", va="top", transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
        )

cbar = fig.colorbar(
    im, ax=axes, orientation='vertical',
    fraction=0.015, pad=0.02,
    ticks=np.arange(0, 1.01, 0.1)
)
cbar.set_label("Perkins Skill Score", fontsize=14)

print(f"PSS range: {vmin:.4f} to {vmax:.4f}")

fig.suptitle("Gridwise Perkins Skill Score on test set (2011-2020)", fontsize=24, fontweight='bold')
plt.savefig(f"{config.OUTPUTS_DIR}/Spatial/gridwise_perkins_skill_score.png", dpi=1000)