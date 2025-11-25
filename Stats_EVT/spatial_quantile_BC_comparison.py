import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import colorcet

bc_methods = ["dOTC", "EQM"]

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

# Validation period: 2011-2023
validation_slice = slice("2011-01-01", "2023-12-31")
obs_tmin_val = xr.open_dataset(obs_tmin_file)["TminD"].sel(time=validation_slice)
obs_tmax_val = xr.open_dataset(obs_tmax_file)["TmaxD"].sel(time=validation_slice)
mask_val = ~np.isnan(obs_tmin_val.isel(time=0).values)

bc_bicubic_tmin_val = {m: xr.open_dataset(bicubic_files_tmin[m])["tmin"].sel(time=validation_slice) for m in bc_methods}
bc_bicubic_tmax_val = {m: xr.open_dataset(bicubic_files_tmax[m])["tmax"].sel(time=validation_slice) for m in bc_methods}

def climatological_cycle(da):
    # Returns DataArrayGroupBy object grouped by dayofyear
    return da.groupby("time.dayofyear")

def gridwise_pss_annual_cycle(a, b, nbins=50):
    # a, b: xarray DataArrayGroupBy objects (grouped by dayofyear)
    doy = np.arange(1, 367)
    shp = a.mean(dim="time").shape  # [lat, lon]
    pss = np.full(shp, np.nan)
    a_group = a
    b_group = b
    for i in range(shp[0]):
        for j in range(shp[1]):
            a_cycle = []
            b_cycle = []
            for d in doy:
                if d in a_group.groups and d in b_group.groups:
                    a_idx = a_group.groups[d]
                    b_idx = b_group.groups[d]
                    a_val = a_group._obj.isel(time=a_idx).values[:, i, j]
                    b_val = b_group._obj.isel(time=b_idx).values[:, i, j]
                    a_cycle.append(np.nanmean(a_val))
                    b_cycle.append(np.nanmean(b_val))
            a_cycle = np.array(a_cycle)
            b_cycle = np.array(b_cycle)
            mask_ij = ~np.isnan(a_cycle) & ~np.isnan(b_cycle)
            if np.sum(mask_ij) > 10:
                a_valid, b_valid = a_cycle[mask_ij], b_cycle[mask_ij]
                combined = np.concatenate([a_valid, b_valid])
                bins = np.linspace(np.min(combined), np.max(combined), nbins + 1)
                hist_a, _ = np.histogram(a_valid, bins=bins, density=True)
                hist_b, _ = np.histogram(b_valid, bins=bins, density=True)
                hist_a /= np.sum(hist_a)
                hist_b /= np.sum(hist_b)
                pss[i, j] = np.sum(np.minimum(hist_a, hist_b))
    return pss

pss_tmin_val = np.stack([
    gridwise_pss_annual_cycle(climatological_cycle(bc_bicubic_tmin_val[m]), climatological_cycle(obs_tmin_val))
    for m in bc_methods
])
pss_tmax_val = np.stack([
    gridwise_pss_annual_cycle(climatological_cycle(bc_bicubic_tmax_val[m]), climatological_cycle(obs_tmax_val))
    for m in bc_methods
])
winner_tmin = np.argmax(pss_tmin_val, axis=0).astype(float)
winner_tmax = np.argmax(pss_tmax_val, axis=0).astype(float)
winner_tmin[~mask_val] = np.nan
winner_tmax[~mask_val] = np.nan

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

all_bc_bias_5_tmin = np.stack([gridwise_percentile_bias(bc_bicubic_tmin[m].values, obs_tmin.values, 5) for m in bc_methods])
all_bc_bias_95_tmax = np.stack([gridwise_percentile_bias(bc_bicubic_tmax[m].values, obs_tmax.values, 95) for m in bc_methods])
all_unet_bias_5_tmin = np.stack([gridwise_percentile_bias(unet_tmin[m].values, obs_tmin.values, 5) for m in bc_methods])
all_unet_bias_95_tmax = np.stack([gridwise_percentile_bias(unet_tmax[m].values, obs_tmax.values, 95) for m in bc_methods])

winner_tmin_int = np.nan_to_num(winner_tmin, nan=-1).astype(int)
winner_tmax_int = np.nan_to_num(winner_tmax, nan=-1).astype(int)

best_bc_bias_5_tmin = np.full(mask.shape, np.nan)
best_bc_bias_95_tmax = np.full(mask.shape, np.nan)
best_unet_bias_5_tmin = np.full(mask.shape, np.nan)
best_unet_bias_95_tmax = np.full(mask.shape, np.nan)

valid_mask = mask & (winner_tmin_int >= 0) & (winner_tmax_int >= 0)
best_bc_bias_5_tmin[valid_mask] = all_bc_bias_5_tmin[winner_tmin_int[valid_mask], np.where(valid_mask)[0], np.where(valid_mask)[1]]
best_bc_bias_95_tmax[valid_mask] = all_bc_bias_95_tmax[winner_tmax_int[valid_mask], np.where(valid_mask)[0], np.where(valid_mask)[1]]
best_unet_bias_5_tmin[valid_mask] = all_unet_bias_5_tmin[winner_tmin_int[valid_mask], np.where(valid_mask)[0], np.where(valid_mask)[1]]
best_unet_bias_95_tmax[valid_mask] = all_unet_bias_95_tmax[winner_tmax_int[valid_mask], np.where(valid_mask)[0], np.where(valid_mask)[1]]

percent_reduction_5_tmin = 100 * (best_bc_bias_5_tmin - best_unet_bias_5_tmin) / np.abs(best_bc_bias_5_tmin)
percent_reduction_95_tmax = 100 * (best_bc_bias_95_tmax - best_unet_bias_95_tmax) / np.abs(best_bc_bias_95_tmax)

fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi=1000)

masked_5_tmin = np.ma.masked_where(~mask, percent_reduction_5_tmin)
masked_95_tmax = np.ma.masked_where(~mask, percent_reduction_95_tmax)

im1 = axs[0].imshow(
    masked_5_tmin,
    origin='lower',
    aspect='auto',
    cmap=colorcet.cm['bwy_r'],
    vmin=-100,
    vmax=100
)
cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04, extend='both')
cbar1.set_label("% Reduction in 5th pctl bias \n(SR+best BC+bicubic over best BC+bicubic)", fontsize=18)
cbar1.ax.tick_params(labelsize=14)
cbar1.ax.set_yticks([-100, -50, 0, 50, 100])
axs[0].set_title("Climatological Mean Tmin: % Reduction in 5th pctl bias", fontsize=18)
axs[0].tick_params(labelsize=16)
axs[0].set_xticks([]); axs[0].set_yticks([])

im2 = axs[1].imshow(
    masked_95_tmax,
    origin='lower',
    aspect='auto',
    cmap=colorcet.cm['bkr_r'],
    vmin=-100,
    vmax=100
)
cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04, extend='both')
cbar2.set_label("% Reduction in 95th pctl bias\n(SR+best BC+bicubic over best BC+bicubic)", fontsize=18)
cbar2.ax.tick_params(labelsize=14)
cbar2.ax.set_yticks([-100, -50, 0, 50, 100])
axs[1].set_title("Climatological Mean Tmax: Reduction in 95th pctl bias", fontsize=18)
axs[1].tick_params(labelsize=16)
axs[1].set_xticks([]); axs[1].set_yticks([])

fig.suptitle("Improvement of SR+best BC+bicubic over best best BC+bicubic\nQuantile Bias Reduction (5th/95th Percentile, 1981–2010)", fontsize=26, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("gridwise_quantile_bias_reduction_SR_best_BC_bicubic_over_best_BC_bicubic_tmin_tmax_5th_95th_poster.png", dpi=1000)
plt.close()