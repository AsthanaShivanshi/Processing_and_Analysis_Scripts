import xarray as xr
import numpy as np
import pandas as pd
from closest_grid_cell import select_nearest_grid_cell
import config
import matplotlib.pyplot as plt

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

percentiles = np.arange(1, 100)  # 1 to 99

obs_ds = xr.open_dataset(obs_precip_file)
time_slice = slice("1981-01-01", "2010-12-31")
obs_precip = obs_ds["RhiresD"].sel(time=time_slice)


quantiles = percentiles / 100  # Convert percentiles to quantiles

for city, (lat, lon) in cities.items():
    plt.figure(figsize=(10, 6), dpi=1000)
    result = select_nearest_grid_cell(obs_ds, lat, lon)
    i, j = result['lat_idx'], result['lon_idx']
    obs_series = obs_precip[:, i, j].values
    obs_pctl = np.nanpercentile(obs_series, percentiles)
    for model in models:
        ds = xr.open_dataset(models[model])
        precip = ds["precip"].sel(time=time_slice)
        result_mod = select_nearest_grid_cell(ds, lat, lon)
        mi_idx, mj_idx = result_mod['lat_idx'], result_mod['lon_idx']
        mod_series = precip[:, mi_idx, mj_idx].values
        mod_pctl = np.nanpercentile(mod_series, percentiles)
        bias_pctl = mod_pctl / obs_pctl
        plt.plot(quantiles, bias_pctl, label=model, linewidth=2)


        plt.axhline(1, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Quantile", fontsize=14)
    plt.ylabel("Multiplicative Bias", fontsize=14)
    plt.title(f"Multiplicative Bias vs Quantile\n{city} (1981–2010)", fontsize=16)
    plt.legend(fontsize=12)
    plt.xticks(np.linspace(0, 1, 11), [str(int(x*100)) for x in np.linspace(0, 1, 11)], fontsize=12)
    plt.tight_layout()
    plt.savefig(f"quantile_bias_{city}_precip.png", dpi=1000)
    plt.close()
