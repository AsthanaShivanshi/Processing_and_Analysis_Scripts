import xarray as xr
import numpy as np
import pandas as pd
from pyextremes import EVA
from closest_grid_cell import select_nearest_grid_cell
import config
import matplotlib.pyplot as plt
import seaborn as sns

dOTC_precip_path = config.BIAS_CORRECTED_DIR + "/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc"
EQM_precip_path = config.BIAS_CORRECTED_DIR + "/EQM/precip_BC_bicubic_r01.nc"
QDM_precip_path = config.BIAS_CORRECTED_DIR + "/QDM/precip_BC_bicubic_r01.nc"
obs_precip_file = config.TARGET_DIR + "/RhiresD_1971_2023.nc"

dOTC_precip_SR= "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/dOTC_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc"
EQM_precip_SR= "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/EQM_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc"
QDM_precip_SR= "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/QDM_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc"

models = {
    "dOTC+bicubic": dOTC_precip_path,
    "dOTC+bicubic+SR": dOTC_precip_SR,
    "EQM+bicubic": EQM_precip_path,
    "EQM+bicubic+SR": EQM_precip_SR,
    "QDM+bicubic": QDM_precip_path,
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
    rolling = np.convolve(arr, np.ones(5), 'valid')
    time_rolled = time_coord[4:]
    years = np.unique(time_rolled.dt.year)
    max_per_year = []
    for year in years:
        mask = time_rolled.dt.year == year
        if np.any(mask):
            max_per_year.append(np.nanmax(rolling[mask]))
        else:
            max_per_year.append(np.nan)
    return pd.Series(max_per_year, index=pd.to_datetime(years, format='%Y'))




def get_return_level(series, periods=[20, 50]):
    eva_model = EVA(series)
    eva_model.get_extremes(method="BM", block_size="365D")
    eva_model.fit_model()
    summary = eva_model.get_summary(return_period=periods, alpha=0.95, n_samples=1000)
    return summary['return value']


bias_20 = np.zeros((len(cities), len(models)))
bias_50 = np.zeros((len(cities), len(models)))
city_names = list(cities.keys())
model_names = list(models.keys())



obs_ds = xr.open_dataset(obs_precip_file)
time_slice = slice("1981-01-01", "2010-12-31")
obs_precip = obs_ds["RhiresD"].sel(time=time_slice)



for ci, (city, (lat, lon)) in enumerate(cities.items()):
    result = select_nearest_grid_cell(obs_ds, lat, lon)
    i, j = result['lat_idx'], result['lon_idx']
    obs_series = annual_max_5day_totals(obs_precip[:, i, j].values, obs_precip["time"])
    obs_rl = get_return_level(obs_series)
    for mi, model in enumerate(model_names):
        ds = xr.open_dataset(models[model])
        precip = ds["precip"].sel(time=time_slice)
        result_mod = select_nearest_grid_cell(ds, lat, lon)
        mi_idx, mj_idx = result_mod['lat_idx'], result_mod['lon_idx']
        mod_series = annual_max_5day_totals(precip[:, mi_idx, mj_idx].values, precip["time"])
        mod_rl = get_return_level(mod_series)
        bias_20[ci, mi] = mod_rl.loc[20] / obs_rl.loc[20]
        bias_50[ci, mi] = mod_rl.loc[50] / obs_rl.loc[50]



plt.figure(figsize=(10, 7), dpi=1000)
sns.heatmap(bias_20, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=model_names, yticklabels=city_names, cbar_kws={'label': 'Multiplicative Bias (20-year RL)'})
plt.title("Multiplicative Bias of 20-Year Return Level for Annual Maximum of 5-Day Total Precipitation\nSwiss Cities (1981–2010)(mm)", fontsize=18)
plt.xlabel("Bias Correction Model", fontsize=14)
plt.ylabel("City", fontsize=14)
plt.tight_layout()
plt.savefig("heatmap_bias_20yrRL_5day_precip_cities_poster.png", dpi=1000)
plt.close()



plt.figure(figsize=(10, 7), dpi=1000)
sns.heatmap(bias_50, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=model_names, yticklabels=city_names, cbar_kws={'label': 'Multiplicative Bias (50-year RL)'})
plt.title("Multiplicative Bias of 50-Year Return Level for Annual Maximum of 5-Day Total Precipitation\nSwiss Cities (1981–2010)(mm)", fontsize=18)
plt.xlabel("Bias Correction Model", fontsize=14)
plt.ylabel("City", fontsize=14)
plt.tight_layout()
plt.savefig("heatmap_bias_50yrRL_5day_precip_cities_poster.png", dpi=1000)
plt.close()