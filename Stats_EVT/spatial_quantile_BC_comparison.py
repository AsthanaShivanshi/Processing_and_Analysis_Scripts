import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import colorcet

bc_methods = ["dOTC", "EQM", "QDM"]

bicubic_files_tmin = {
    "dOTC": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc",
    "EQM":  "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/EQM/tmin_BC_bicubic_r01.nc",
    "QDM":  "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/QDM/tmin_BC_bicubic_r01.nc"
}
bicubic_files_tmax = {
    "dOTC": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc",
    "EQM":  "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/EQM/tmax_BC_bicubic_r01.nc",
    "QDM":  "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/BC_Model_Runs/QDM/tmax_BC_bicubic_r01.nc"
}
unet_files_tmin = {
    "dOTC": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/dOTC_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc",
    "EQM":  "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/EQM_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc",
    "QDM":  "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/QDM_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc"
}
unet_files_tmax = unet_files_tmin  



obs_tmin_file = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/TminD_1971_2023.nc"
obs_tmax_file = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/TmaxD_1971_2023.nc"
time_slice = slice("1981-01-01", "2010-12-31")



obs_tmin = xr.open_dataset(obs_tmin_file)["TminD"].sel(time=time_slice)
obs_tmax = xr.open_dataset(obs_tmax_file)["TmaxD"].sel(time=time_slice)
mask = ~np.isnan(obs_tmin.isel(time=0).values)



# Load BC+bicubic and UNet files
bc_bicubic_tmin = {m: xr.open_dataset(bicubic_files_tmin[m])["tmin"].sel(time=time_slice) for m in bc_methods}
bc_bicubic_tmax = {m: xr.open_dataset(bicubic_files_tmax[m])["tmax"].sel(time=time_slice) for m in bc_methods}
unet_tmin = {m: xr.open_dataset(unet_files_tmin[m])["tmin"].sel(time=time_slice) for m in bc_methods}
unet_tmax = {m: xr.open_dataset(unet_files_tmax[m])["tmax"].sel(time=time_slice) for m in bc_methods}



def gridwise_pss(a, b, nbins=50):
    # Vectorized version
    shp = a.shape[1:]
    pss = np.full(shp, np.nan)
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    for idx in range(a_flat.shape[1]):
        a1, b1 = a_flat[:, idx], b_flat[:, idx]
        mask_ij = ~np.isnan(a1) & ~np.isnan(b1)
        if np.sum(mask_ij) > 10:
            a_valid, b_valid = a1[mask_ij], b1[mask_ij]
            combined = np.concatenate([a_valid, b_valid])
            bins = np.linspace(np.min(combined), np.max(combined), nbins + 1)
            hist_a, _ = np.histogram(a_valid, bins=bins, density=True)
            hist_b, _ = np.histogram(b_valid, bins=bins, density=True)
            hist_a /= np.sum(hist_a)
            hist_b /= np.sum(hist_b)
            pss.flat[idx] = np.sum(np.minimum(hist_a, hist_b))
    return pss




def gridwise_percentile_bias(a, b, percentile):
    # Vectorized version
    shp = a.shape[1:]
    bias = np.full(shp, np.nan)
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    for idx in range(a_flat.shape[1]):
        a1, b1 = a_flat[:, idx], b_flat[:, idx]
        mask_ij = ~np.isnan(a1) & ~np.isnan(b1)
        if np.sum(mask_ij) > 10:
            bias.flat[idx] = np.nanpercentile(a1[mask_ij], percentile) - np.nanpercentile(b1[mask_ij], percentile)
    return bias



# Best BC method per grid cell based on PSS
pss_tmin = np.stack([gridwise_pss(bc_bicubic_tmin[m].values, obs_tmin.values) for m in bc_methods])
pss_tmax = np.stack([gridwise_pss(bc_bicubic_tmax[m].values, obs_tmax.values) for m in bc_methods])
winner_tmin = np.argmax(pss_tmin, axis=0).astype(float)
winner_tmax = np.argmax(pss_tmax, axis=0).astype(float)
winner_tmin[~mask] = np.nan
winner_tmax[~mask] = np.nan



# Precompute all biases for all methods
all_bc_bias_10_tmin = np.stack([gridwise_percentile_bias(bc_bicubic_tmin[m].values, obs_tmin.values, 10) for m in bc_methods])
all_bc_bias_90_tmax = np.stack([gridwise_percentile_bias(bc_bicubic_tmax[m].values, obs_tmax.values, 90) for m in bc_methods])
all_unet_bias_10_tmin = np.stack([gridwise_percentile_bias(unet_tmin[m].values, obs_tmin.values, 10) for m in bc_methods])
all_unet_bias_90_tmax = np.stack([gridwise_percentile_bias(unet_tmax[m].values, obs_tmax.values, 90) for m in bc_methods])



# Select best method per cell using winner_tmin and winner_tmax
winner_tmin_int = np.nan_to_num(winner_tmin, nan=-1).astype(int)
winner_tmax_int = np.nan_to_num(winner_tmax, nan=-1).astype(int)


best_bc_bias_10_tmin = np.full(mask.shape, np.nan)
best_bc_bias_90_tmax = np.full(mask.shape, np.nan)
best_unet_bias_10_tmin = np.full(mask.shape, np.nan)
best_unet_bias_90_tmax = np.full(mask.shape, np.nan)

valid_mask = mask & (winner_tmin_int >= 0) & (winner_tmax_int >= 0)
best_bc_bias_10_tmin[valid_mask] = all_bc_bias_10_tmin[winner_tmin_int[valid_mask], np.where(valid_mask)[0], np.where(valid_mask)[1]]
best_bc_bias_90_tmax[valid_mask] = all_bc_bias_90_tmax[winner_tmax_int[valid_mask], np.where(valid_mask)[0], np.where(valid_mask)[1]]
best_unet_bias_10_tmin[valid_mask] = all_unet_bias_10_tmin[winner_tmin_int[valid_mask], np.where(valid_mask)[0], np.where(valid_mask)[1]]
best_unet_bias_90_tmax[valid_mask] = all_unet_bias_90_tmax[winner_tmax_int[valid_mask], np.where(valid_mask)[0], np.where(valid_mask)[1]]

percent_reduction_10_tmin = 100 * (best_bc_bias_10_tmin - best_unet_bias_10_tmin) / np.abs(best_bc_bias_10_tmin)
percent_reduction_90_tmax = 100 * (best_bc_bias_90_tmax - best_unet_bias_90_tmax) / np.abs(best_bc_bias_90_tmax)

fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi=300)




masked_10_tmin = np.ma.masked_where(~mask, percent_reduction_10_tmin)
masked_90_tmax = np.ma.masked_where(~mask, percent_reduction_90_tmax)



#Tmin 10th percentile bias reduction
im1 = axs[0].imshow(masked_10_tmin, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-100, vmax=100)
cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04, extend='both')
cbar1.set_label("% Reduction in 10th Percentile Bias\n(SR+BC+bicubic vs BC+bicubic)", fontsize=18)
cbar1.ax.tick_params(labelsize=14)
cbar1.ax.set_yticks([-100, -50, 0, 50, 100])
axs[0].set_title("Tmin: % Reduction in 10th Percentile Bias", fontsize=22, fontweight='bold')
axs[0].tick_params(labelsize=16)
axs[0].set_xticks([]); axs[0].set_yticks([])

#Tmax 90th percentile bias reduction
im2 = axs[1].imshow(masked_90_tmax, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-100, vmax=100)
cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04, extend='both')
cbar2.set_label("% Reduction in 90th Percentile Bias\n(SR+BC+bicubic vs BC+bicubic)", fontsize=18)
cbar2.ax.tick_params(labelsize=14)
cbar2.ax.set_yticks([-100, -50, 0, 50, 100])
axs[1].set_title("Tmax: % Reduction in 90th Percentile Bias", fontsize=22, fontweight='bold')
axs[1].tick_params(labelsize=16)
axs[1].set_xticks([]); axs[1].set_yticks([])


fig.suptitle("Spatial Improvement of SR+BC+bicubic over BC+bicubic\nQuantile Bias Reduction (1981–2010)", fontsize=26, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("gridwise_quantile_bias_reduction_SR_BC_bicubic_over_BC_bicubic_tmin_tmax_poster.png", dpi=1000)
plt.close()