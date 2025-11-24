import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

bc_bicubic_tmin = {m: xr.open_dataset(bicubic_files_tmin[m])["tmin"].sel(time=time_slice) for m in bc_methods}
bc_bicubic_tmax = {m: xr.open_dataset(bicubic_files_tmax[m])["tmax"].sel(time=time_slice) for m in bc_methods}
unet_tmin = {m: xr.open_dataset(unet_files_tmin[m])["tmin"].sel(time=time_slice) for m in bc_methods}
unet_tmax = {m: xr.open_dataset(unet_files_tmax[m])["tmax"].sel(time=time_slice) for m in bc_methods}

def gridwise_pss(a, b, nbins=50):
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

# Five cities and their coordinates
cities = {
    "Bern": (46.9480, 7.4474),
    "Geneva": (46.2044, 6.1432),
    "Locarno": (46.1709, 8.7995),
    "Lugano": (46.0037, 8.9511),
    "Zürich": (47.3769, 8.5417)
}

def find_closest_grid(ds, lat, lon):
    lat_arr = ds['lat'].values if 'lat' in ds else ds['latitude'].values
    lon_arr = ds['lon'].values if 'lon' in ds else ds['longitude'].values
    lat_idx = np.abs(lat_arr - lat).argmin()
    lon_idx = np.abs(lon_arr - lon).argmin()
    return lat_idx, lon_idx



# Compute PSS for all methods
pss_tmin = np.stack([gridwise_pss(bc_bicubic_tmin[m].values, obs_tmin.values) for m in bc_methods])
pss_tmax = np.stack([gridwise_pss(bc_bicubic_tmax[m].values, obs_tmax.values) for m in bc_methods])
winner_tmin = np.argmax(pss_tmin, axis=0)
winner_tmax = np.argmax(pss_tmax, axis=0)



# Compute percentile biases for all methods
all_bc_bias_10_tmin = np.stack([gridwise_percentile_bias(bc_bicubic_tmin[m].values, obs_tmin.values, 10) for m in bc_methods])
all_bc_bias_90_tmax = np.stack([gridwise_percentile_bias(bc_bicubic_tmax[m].values, obs_tmax.values, 90) for m in bc_methods])
all_unet_bias_10_tmin = np.stack([gridwise_percentile_bias(unet_tmin[m].values, obs_tmin.values, 10) for m in bc_methods])
all_unet_bias_90_tmax = np.stack([gridwise_percentile_bias(unet_tmax[m].values, obs_tmax.values, 90) for m in bc_methods])



city_names = []
tmin_bias_reduction = []
tmax_bias_reduction = []

for city, (lat, lon) in cities.items():
    # Use obs_tmin for grid finding
    lat_idx, lon_idx = find_closest_grid(obs_tmin, lat, lon)
    # Get best method index for this grid cell
    best_tmin_method = winner_tmin[lat_idx, lon_idx]
    best_tmax_method = winner_tmax[lat_idx, lon_idx]
    # Get biases for BC+bicubic and UNet for best method
    bc_bias_10_tmin = all_bc_bias_10_tmin[best_tmin_method, lat_idx, lon_idx]
    unet_bias_10_tmin = all_unet_bias_10_tmin[best_tmin_method, lat_idx, lon_idx]
    bc_bias_90_tmax = all_bc_bias_90_tmax[best_tmax_method, lat_idx, lon_idx]
    unet_bias_90_tmax = all_unet_bias_90_tmax[best_tmax_method, lat_idx, lon_idx]
    # Compute percent reduction
    reduction_10_tmin = 100 * (bc_bias_10_tmin - unet_bias_10_tmin) / np.abs(bc_bias_10_tmin) if np.abs(bc_bias_10_tmin) > 0 else np.nan
    reduction_90_tmax = 100 * (bc_bias_90_tmax - unet_bias_90_tmax) / np.abs(bc_bias_90_tmax) if np.abs(bc_bias_90_tmax) > 0 else np.nan
    city_names.append(city)
    tmin_bias_reduction.append(reduction_10_tmin)
    tmax_bias_reduction.append(reduction_90_tmax)




df = pd.DataFrame({
    "City": city_names,
    "Tmin 10th %ile Bias Reduction over Best BC+bicubic": tmin_bias_reduction,
    "Tmax 90th %ile Bias Reduction over Best BC+bicubic": tmax_bias_reduction
})
df.set_index("City", inplace=True)
heatmap_data = df.T

plt.figure(figsize=(8, 4), dpi=300)
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="coolwarm", center=0, cbar_kws={'label': '% Bias Reduction'})
plt.title("% Bias Reduction for Cities\nTmin 10th and Tmax 90th Percentile (UNet vs BC+bicubic)")
plt.ylabel("Metric")
plt.xlabel("City")
plt.tight_layout()
plt.savefig("city_heatmap_bias_reduction_tmin10_tmax90.png", dpi=300)
plt.close()