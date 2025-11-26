import xarray as xr
import numpy as np
import pandas as pd
from closest_grid_cell import select_nearest_grid_cell
import config
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

dOTC_precip_path = config.BIAS_CORRECTED_DIR + "/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc"
EQM_precip_path = config.BIAS_CORRECTED_DIR + "/EQM/precip_BC_bicubic_r01.nc"
obs_precip_file = config.TARGET_DIR + "/RhiresD_1971_2023.nc"

dOTC_precip_SR= "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/dOTC_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc"
EQM_precip_SR= "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/EQM_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc"

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

def rolling_5day_totals(arr):
    return np.convolve(arr, np.ones(5), 'valid')

city_names = list(cities.keys())
model_names = list(models.keys())

obs_ds = xr.open_dataset(obs_precip_file)
time_slice = slice("1981-01-01", "2010-12-31")
obs_precip = obs_ds["RhiresD"].sel(time=time_slice)

percentile = 99

data = []
for ci, (city, (lat, lon)) in enumerate(cities.items()):
    result = select_nearest_grid_cell(obs_ds, lat, lon)
    i, j = result['lat_idx'], result['lon_idx']
    obs_rolling = rolling_5day_totals(obs_precip[:, i, j].values)
    obs_pctl = np.nanpercentile(obs_rolling, percentile)
    for mi, model in enumerate(model_names):
        ds = xr.open_dataset(models[model])
        precip = ds["precip"].sel(time=time_slice)
        result_mod = select_nearest_grid_cell(ds, lat, lon)
        mi_idx, mj_idx = result_mod['lat_idx'], result_mod['lon_idx']
        mod_rolling = rolling_5day_totals(precip[:, mi_idx, mj_idx].values)
        mod_pctl = np.nanpercentile(mod_rolling, percentile)
        bias = mod_pctl / obs_pctl
        data.append({
            "City": city,
            "Model": model,
            "Obs_99th": obs_pctl,
            "Model_99th": mod_pctl,
            "Bias": bias
        })

df = pd.DataFrame(data)

palette = [
    "#0072B2", "#D55E00", "#009E73", "#F0E442", "#CC79A7", "#56B4E9"
]
model_palette = {model: palette[i % len(palette)] for i, model in enumerate(model_names)}

city_markers = {
    "Bern": "o",
    "Geneva": "s",
    "Locarno": "D",
    "Lugano": "^",
    "Zürich": "P"
}

plt.figure(figsize=(12, 7), dpi=1000)
ax = sns.scatterplot(
    data=df,
    x="City",
    y="Bias",
    hue="Model",
    palette=model_palette,
    style="City",
    markers=city_markers,
    s=120,
    legend=False
)
plt.axhline(1, color='black', linestyle='--', linewidth=2, label='No Bias (y=1)')
plt.ylabel("Multiplicative Bias (99th percentile 5-day total)", fontsize=14)
plt.title("Multiplicative Bias of 99th Percentile\n5-Day Total Precipitation (1981–2010)", fontsize=18)
plt.xticks(range(len(city_names)), city_names, fontsize=13)
plt.tight_layout()
for i in range(1, len(city_names)):
    plt.axvline(i - 0.5, color='gray', linestyle=':', linewidth=1)

handles = [Patch(facecolor=model_palette[model], edgecolor='black', label=model) for model in model_names]
ax.legend(handles=handles, title="Model", loc='upper left', bbox_to_anchor=(1,1), fontsize=12, title_fontsize=13)

plt.savefig("dotplot_bias_99th_5day_precip_cities.png", dpi=1000, bbox_inches="tight")
plt.close()