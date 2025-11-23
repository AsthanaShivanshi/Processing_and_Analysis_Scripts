import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
from closest_grid_cell import select_nearest_grid_cell
from scores.continuous import multiplicative_bias
from scores.continuous.correlation import pearsonr
from scores.continuous import kge


sns.set(style="whitegrid")

# File paths
dOTC_precip_path = config.BIAS_CORRECTED_DIR + "/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc"
EQM_precip_path = config.BIAS_CORRECTED_DIR + "/EQM/precip_BC_bicubic_r01.nc"
QDM_precip_path = config.BIAS_CORRECTED_DIR + "/QDM/precip_BC_bicubic_r01.nc"
obs_precip_file = config.TARGET_DIR + "/RhiresD_1971_2023.nc"

dOTC_precip_SR= "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/dOTC_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc"
EQM_precip_SR= "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/EQM_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc"
QDM_precip_SR= "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/QDM_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc"

# Load datasets
dOTC_ds_bicubic = xr.open_dataset(dOTC_precip_path)
EQM_ds_bicubic = xr.open_dataset(EQM_precip_path)
QDM_ds_bicubic = xr.open_dataset(QDM_precip_path)
obs_ds = xr.open_dataset(obs_precip_file)

dOTC_ds_SR = xr.open_dataset(dOTC_precip_SR)
EQM_ds_SR = xr.open_dataset(EQM_precip_SR)
QDM_ds_SR = xr.open_dataset(QDM_precip_SR)

time_slice = slice("1981-01-01", "2010-12-31")

dOTC_precip = dOTC_ds_bicubic["precip"].sel(time=time_slice)
EQM_precip = EQM_ds_bicubic["precip"].sel(time=time_slice)
QDM_precip = QDM_ds_bicubic["precip"].sel(time=time_slice)
obs_precip = obs_ds["RhiresD"].sel(time=time_slice)

dOTC_precip_SR = dOTC_ds_SR["precip"].sel(time=time_slice)
EQM_precip_SR = EQM_ds_SR["precip"].sel(time=time_slice)
QDM_precip_SR = QDM_ds_SR["precip"].sel(time=time_slice)





models = {
    "dOTC+bicubic": dOTC_precip,
    "dOTC+bicubic+SR": dOTC_precip_SR,
    "EQM+bicubic": EQM_precip,
    "EQM+bicubic+SR": EQM_precip_SR,
    "QDM+bicubic": QDM_precip,
    "QDM+bicubic+SR": QDM_precip_SR,
}

cities = {
    "Bern": (46.9480, 7.4474),
    "Geneva": (46.2044, 6.1432),
    "Locarno": (46.1709, 8.7995),
    "Lugano": (46.0037, 8.9511),
    "Zürich": (47.3769, 8.5417)
}



def annual_max_5day_totals(arr, time_coord):
    # arr: (time,)
    # time_coord: xarray time coordinate
    # Returns: array of annual maxima, years aligned with obs
    rolling = np.convolve(arr, np.ones(5), 'valid')
    # Align time after rolling
    time_rolled = time_coord[4:]
    years = np.unique(time_rolled.dt.year)
    max_per_year = []
    for year in years:
        mask = time_rolled.dt.year == year
        if np.any(mask):
            max_per_year.append(np.nanmax(rolling[mask]))
        else:
            max_per_year.append(np.nan)
    return np.array(max_per_year), years



# KGE matrix
heatmap_data = np.zeros((len(cities), len(models)))
city_names = list(cities.keys())
model_names = list(models.keys())

for ci, (city, (lat, lon)) in enumerate(cities.items()):
    result = select_nearest_grid_cell(obs_ds, lat, lon)
    i, j = result['lat_idx'], result['lon_idx']
    obs_max, years = annual_max_5day_totals(obs_precip[:, i, j].values, obs_precip["time"])
    print(f"\nCity: {city}")
    for mi, model in enumerate(model_names):
        mod_max, mod_years = annual_max_5day_totals(models[model][:, i, j].values, models[model]["time"])
        # Align years
        common_years = np.intersect1d(years, mod_years)
        obs_max_aligned = obs_max[np.isin(years, common_years)]
        mod_max_aligned = mod_max[np.isin(mod_years, common_years)]
        # Wrap as xarray.DataArray
        mod_max_da = xr.DataArray(mod_max_aligned)
        obs_max_da = xr.DataArray(obs_max_aligned)
        kge_result = kge(mod_max_da, obs_max_da, include_components=True)
        heatmap_data[ci, mi] = kge_result['kge'].item()
        print(f"{model}: KGE = {kge_result['kge'].item():.3f}, "
              f"correlation(rho) = {kge_result['rho'].item():.3f}, "
              f"variability(alpha) = {kge_result['alpha'].item():.3f}, "
              f"bias(beta) = {kge_result['beta'].item():.3f}")


# Plot heatmap
plt.figure(figsize=(10, 7), dpi=1000)
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=model_names, yticklabels=city_names, cbar_kws={'label': 'Kling-Gupta Efficiency'})
plt.title("Kling-Gupta Efficiency for Annual Maximum of 5-Day Precipitation Totals\nSwiss Cities (1981–2010)", fontsize=18, pad=15)
plt.xlabel("Bias Correction Model", fontsize=14)
plt.ylabel("City", fontsize=14)
plt.tight_layout()
plt.savefig("heatmap_KGE_5day_precip_cities_poster.png", dpi=1000)
plt.close()