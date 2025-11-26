import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import config

bc_methods = ["dOTC", "EQM"]

np.Inf=np.inf

bicubic_files_tmin = {
    "dOTC": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc",
    "EQM":  "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/EQM/tmin_BC_bicubic_r01.nc",
}
bicubic_files_tmax = {
    "dOTC": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc",
    "EQM":  "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/EQM/tmax_BC_bicubic_r01.nc",
}
unet_files_tmin = {
    "dOTC": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/dOTC_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc",
    "EQM":  "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/EQM_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc",
}
unet_files_tmax = unet_files_tmin  

obs_tmin_file = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/TminD_1971_2023.nc"
obs_tmax_file = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/TmaxD_1971_2023.nc"

# Analysis period: 1981-2010
analysis_slice = slice("1981-01-01", "2010-12-31")
obs_tmin = xr.open_dataset(obs_tmin_file)["TminD"].sel(time=analysis_slice)
obs_tmax = xr.open_dataset(obs_tmax_file)["TmaxD"].sel(time=analysis_slice)
mask = ~np.isnan(obs_tmin.isel(time=0).values)

bc_bicubic_tmin = {m: xr.open_dataset(bicubic_files_tmin[m])["tmin"].sel(time=analysis_slice) for m in bc_methods}
bc_bicubic_tmax = {m: xr.open_dataset(bicubic_files_tmax[m])["tmax"].sel(time=analysis_slice) for m in bc_methods}
unet_tmin = {m: xr.open_dataset(unet_files_tmin[m])["tmin"].sel(time=analysis_slice) for m in bc_methods}
unet_tmax = {m: xr.open_dataset(unet_files_tmax[m])["tmax"].sel(time=analysis_slice) for m in bc_methods}

def gridwise_percentile_bias(a, b, percentile):
    a_pctl = np.nanpercentile(a, percentile, axis=0)
    b_pctl = np.nanpercentile(b, percentile, axis=0)
    bias = a_pctl - b_pctl
    valid_counts = np.sum(~np.isnan(a) & ~np.isnan(b), axis=0)
    bias[valid_counts <= 10] = np.nan
    return bias

# Compute bias for each BC method
all_bc_bias_5_tmin = np.stack([gridwise_percentile_bias(bc_bicubic_tmin[m].values, obs_tmin.values, 5) for m in bc_methods])
all_bc_bias_95_tmax = np.stack([gridwise_percentile_bias(bc_bicubic_tmax[m].values, obs_tmax.values, 95) for m in bc_methods])
all_unet_bias_5_tmin = np.stack([gridwise_percentile_bias(unet_tmin[m].values, obs_tmin.values, 5) for m in bc_methods])
all_unet_bias_95_tmax = np.stack([gridwise_percentile_bias(unet_tmax[m].values, obs_tmax.values, 95) for m in bc_methods])


best_bc_bias_5_tmin = np.full(mask.shape, np.nan)
best_bc_bias_95_tmax = np.full(mask.shape, np.nan)
best_unet_bias_5_tmin = np.full(mask.shape, np.nan)
best_unet_bias_95_tmax = np.full(mask.shape, np.nan)

# Mask for valid grid cells (at least one non-NaN bias for each case)
valid_bc_5_tmin = np.any(~np.isnan(all_bc_bias_5_tmin), axis=0)
valid_bc_95_tmax = np.any(~np.isnan(all_bc_bias_95_tmax), axis=0)

# Set all-NaN slices to a large value so they won't be chosen
all_bc_bias_5_tmin_filled = np.where(np.isnan(all_bc_bias_5_tmin), np.inf, np.abs(all_bc_bias_5_tmin))
all_bc_bias_95_tmax_filled = np.where(np.isnan(all_bc_bias_95_tmax), np.inf, np.abs(all_bc_bias_95_tmax))

best_bc_idx_5_tmin = np.argmin(all_bc_bias_5_tmin_filled, axis=0)
best_bc_idx_95_tmax = np.argmin(all_bc_bias_95_tmax_filled, axis=0)

# Only assign for valid grid cells
best_bc_bias_5_tmin = np.full(mask.shape, np.nan)
best_bc_bias_95_tmax = np.full(mask.shape, np.nan)
best_unet_bias_5_tmin = np.full(mask.shape, np.nan)
best_unet_bias_95_tmax = np.full(mask.shape, np.nan)

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if mask[i, j] and valid_bc_5_tmin[i, j]:
            idx5 = best_bc_idx_5_tmin[i, j]
            best_bc_bias_5_tmin[i, j] = all_bc_bias_5_tmin[idx5, i, j]
            best_unet_bias_5_tmin[i, j] = all_unet_bias_5_tmin[idx5, i, j]
        if mask[i, j] and valid_bc_95_tmax[i, j]:
            idx95 = best_bc_idx_95_tmax[i, j]
            best_bc_bias_95_tmax[i, j] = all_bc_bias_95_tmax[idx95, i, j]
            best_unet_bias_95_tmax[i, j] = all_unet_bias_95_tmax[idx95, i, j]


# Binary improvement maps
improvement_5_tmin = np.zeros(mask.shape, dtype=int)
improvement_95_tmax = np.zeros(mask.shape, dtype=int)

valid_mask_5_tmin = mask & ~np.isnan(best_bc_bias_5_tmin) & ~np.isnan(best_unet_bias_5_tmin)
valid_mask_95_tmax = mask & ~np.isnan(best_bc_bias_95_tmax) & ~np.isnan(best_unet_bias_95_tmax)

improvement_5_tmin[valid_mask_5_tmin] = (
    np.abs(best_unet_bias_5_tmin[valid_mask_5_tmin]) < np.abs(best_bc_bias_5_tmin[valid_mask_5_tmin])
).astype(int)
improvement_95_tmax[valid_mask_95_tmax] = (
    np.abs(best_unet_bias_95_tmax[valid_mask_95_tmax]) < np.abs(best_bc_bias_95_tmax[valid_mask_95_tmax])
).astype(int)

# Calculate percentage of grid cells with improvement
percent_improved_5_tmin = 100 * np.sum(improvement_5_tmin) / np.sum(valid_mask_5_tmin)
percent_improved_95_tmax = 100 * np.sum(improvement_95_tmax) / np.sum(valid_mask_95_tmax)


fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi=1000)

im1 = axs[0].imshow(
    np.ma.masked_where(~mask, improvement_5_tmin),
    origin='lower',
    aspect='auto',
    cmap='magma',  
    vmin=0,
    vmax=1
)
cbar1 = fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
cbar1.outline.set_edgecolor('black')
cbar1.ax.tick_params(colors='black')
cbar1.set_label('Improvement', fontsize=18, color='black')

axs[0].set_title(f"Tmin 5th Percentile Bias Improvement\n{percent_improved_5_tmin:.1f}% grid cells improved", fontsize=18)
axs[0].tick_params(labelsize=18)
axs[0].set_xticks([]); axs[0].set_yticks([])

im2 = axs[1].imshow(
    np.ma.masked_where(~mask, improvement_95_tmax),
    origin='lower',
    aspect='auto',
    cmap='inferno', 
    vmin=0,
    vmax=1
)
cbar2 = fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
cbar2.outline.set_edgecolor('black')
cbar2.ax.tick_params(colors='black')
cbar2.set_label('Improvement', fontsize=18, color='black')

axs[1].set_title(f"Tmax 95th Percentile Bias Improvement\n{percent_improved_95_tmax:.1f}% grid cells improved", fontsize=20)
axs[1].tick_params(labelsize=18)
axs[1].set_xticks([]); axs[1].set_yticks([])

fig.suptitle("Gridwise Bias Improvement: SR+best BC+bicubic vs best BC+bicubic\n(5th/95th Percentile, 1981–2010)", fontsize=30, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("gridwise_quantile_bias_improvement_SR_best_BC_bicubic_over_best_BC_bicubic_tmin_tmax_5th_95th_binary.png", dpi=1000)
plt.close()
