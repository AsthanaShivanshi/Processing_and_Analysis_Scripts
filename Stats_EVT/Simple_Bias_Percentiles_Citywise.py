import xarray as xr
import numpy as np
import pandas as pd
from closest_grid_cell import select_nearest_grid_cell
import config
import matplotlib.pyplot as plt


np.Inf= np.inf

def rolling_5day_totals(arr):
    arr = np.array(arr)
    totals = np.full(arr.shape, np.nan)
    if arr.size >= 5:
        totals[4:] = np.convolve(arr, np.ones(5), 'valid')
    return totals

# File paths
dOTC_precip_path = config.BIAS_CORRECTED_DIR + "/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc"
EQM_precip_path = config.BIAS_CORRECTED_DIR + "/EQM/precip_BC_bicubic_r01.nc"
obs_precip_file = config.TARGET_DIR + "/RhiresD_1971_2023.nc"
dOTC_precip_SR = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/dOTC_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc"
EQM_precip_SR = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/EQM_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc"

models = {
    "dOTC+bicubic": dOTC_precip_path,
    "dOTC+bicubic+SR": dOTC_precip_SR,
    "EQM+bicubic": EQM_precip_path,
    "EQM+bicubic+SR": EQM_precip_SR,
}

cities = {
    "Bern": (46.9480, 7.4474),
    "Geneva": (46.2044, 6.1432),
    "Locarno": (46.1709, 8.7995),
    "Lugano": (46.0037, 8.9511),
    "Zürich": (47.3769, 8.5417)
}

percentiles = np.arange(1, 100)
quantiles = percentiles / 100

obs_ds = xr.open_dataset(obs_precip_file)
time_slice = slice("1981-01-01", "2010-12-31")
obs_precip = obs_ds["RhiresD"].sel(time=time_slice)

for city, (lat, lon) in cities.items():
    plt.figure(figsize=(10, 6), dpi=300)
    result = select_nearest_grid_cell(obs_ds, lat, lon)
    i, j = result['lat_idx'], result['lon_idx']
    obs_series = obs_precip[:, i, j].values
    obs_5day = rolling_5day_totals(obs_series)
    obs_qtls = np.nanquantile(obs_5day, quantiles)

    for model in models:
        ds = xr.open_dataset(models[model])
        precip = ds["precip"].sel(time=time_slice)
        result_mod = select_nearest_grid_cell(ds, lat, lon)
        mi_idx, mj_idx = result_mod['lat_idx'], result_mod['lon_idx']
        mod_series = precip[:, mi_idx, mj_idx].values
        mod_5day = rolling_5day_totals(mod_series)
        mod_qtls = np.nanquantile(mod_5day, quantiles)
        bias_qtls = mod_qtls - obs_qtls
        plt.plot(quantiles, bias_qtls, label=model, linewidth=2)

    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Quantile", fontsize=14)
    plt.ylabel("Quantile Bias (5-day totals)", fontsize=14)
    plt.title(f"Bias vs Quantile\n{city} (5-day Precip Totals, 1981–2010)", fontsize=16)
    plt.legend(fontsize=12)
    plt.xticks(np.linspace(0, 1, 10), [str(int(x*100)) for x in np.linspace(0, 0.9, 10)], fontsize=12)  # Only up to 0.9 (90%)
    plt.tight_layout()
    plt.savefig(f"quantile_bias_{city}_5day_precip.png", dpi=300)
    plt.close()
