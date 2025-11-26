import xarray as xr
import numpy as np
import pandas as pd
from pyextremes import EVA
from closest_grid_cell import select_nearest_grid_cell
import config
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch

dOTC_precip_path = config.BIAS_CORRECTED_DIR + "/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc"
EQM_precip_path = config.BIAS_CORRECTED_DIR + "/EQM/precip_BC_bicubic_r01.nc"
# QDM_precip_path = config.BIAS_CORRECTED_DIR + "/QDM/precip_BC_bicubic_r01.nc"  # Removed
obs_precip_file = config.TARGET_DIR + "/RhiresD_1971_2023.nc"

dOTC_precip_SR= "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/dOTC_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc"
EQM_precip_SR= "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/EQM_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc"
# QDM_precip_SR= "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/UNet_Deterministic_Training_Dataset/QDM_ModelRun_Downscaled_Predictions_Validation_1981_2010.nc"  # Removed

models = {
    "dOTC+bicubic": dOTC_precip_path,
    "dOTC+bicubic+SR": dOTC_precip_SR,
    "EQM+bicubic": EQM_precip_path,
    "EQM+bicubic+SR": EQM_precip_SR,
    # "QDM+bicubic": QDM_precip_path,  # Removed
    # "QDM+bicubic+SR": QDM_precip_SR,  # Removed
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

def get_return_level(series, periods=[50, 100]):
    eva_model = EVA(series)
    eva_model.get_extremes(method="BM", block_size="365D")
    eva_model.fit_model()
    summary = eva_model.get_summary(return_period=periods, alpha=0.95, n_samples=1000)
    return summary['return value']

bias_50 = np.zeros((len(cities), len(models)))
bias_100 = np.zeros((len(cities), len(models)))
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
        bias_50[ci, mi] = mod_rl.loc[50] / obs_rl.loc[50]
        bias_100[ci, mi] = mod_rl.loc[100] / obs_rl.loc[100]

# Prepare data for dot plots
data = []
for ci, city in enumerate(city_names):
    for mi, model in enumerate(model_names):
        data.append({"City": city, "Model": model, "Period": "50-year RL", "Bias": bias_50[ci, mi]})
        data.append({"City": city, "Model": model, "Period": "100-year RL", "Bias": bias_100[ci, mi]})
df = pd.DataFrame(data)

# Two different colorblind-friendly palettes for the two plots
palette_50 = [
    "#0072B2", "#D55E00", "#009E73", "#F0E442", "#CC79A7", "#56B4E9"
]

palette_100 = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"
]
model_palette_50 = {model: palette_50[i % len(palette_50)] for i, model in enumerate(model_names)}
model_palette_100 = {model: palette_100[i % len(palette_100)] for i, model in enumerate(model_names)}


# Define marker shapes for each city
city_markers = {
    "Bern": "o",
    "Geneva": "s",
    "Locarno": "D",
    "Lugano": "^",
    "Zürich": "P"
}

# Dot plot for 50-year RL
plt.figure(figsize=(12, 7), dpi=1000)
ax = sns.scatterplot(
    data=df[df["Period"] == "50-year RL"],
    x="City",
    y="Bias",
    hue="Model",
    palette=model_palette_50,
    style="City",
    markers=city_markers,
    s=120,
    legend=False  # Disable automatic legend
)
plt.axhline(1, color='black', linestyle='--', linewidth=2, label='No Bias (y=1)')
plt.ylabel("Multiplicative Bias (50-year RL)", fontsize=14)
plt.title("Multiplicative Bias of 50-Year Return Level\nAnnual Max 5-Day Total Precipitation (1981–2010)", fontsize=18)
plt.xticks(range(len(city_names)), city_names, fontsize=13)
plt.tight_layout()
for i in range(1, len(city_names)):
    plt.axvline(i - 0.5, color='gray', linestyle=':', linewidth=1)

# Custom legend: colored rectangles for models
handles = [Patch(facecolor=model_palette_50[model], edgecolor='black', label=model) for model in model_names]
ax.legend(handles=handles, title="Model", loc='upper left', bbox_to_anchor=(1,1), fontsize=12, title_fontsize=13)

plt.savefig("dotplot_bias_50yrRL_5day_precip_cities_poster.png", dpi=1000, bbox_inches="tight")
plt.close()

# Dot plot for 100-year RL
plt.figure(figsize=(12, 7), dpi=1000)
ax = sns.scatterplot(
    data=df[df["Period"] == "100-year RL"],
    x="City",
    y="Bias",
    hue="Model",
    palette=model_palette_100,
    style="City",
    markers=city_markers,
    s=120,
    legend=False  # Disable automatic legend
)
plt.axhline(1, color='black', linestyle='--', linewidth=2, label='No Bias (y=1)')
plt.ylabel("Multiplicative Bias (100-year RL)", fontsize=14)
plt.title("Multiplicative Bias of 100-Year Return Level\nAnnual Max 5-Day Total Precipitation (1981–2010)", fontsize=18)
plt.xticks(range(len(city_names)), city_names, fontsize=13)
plt.tight_layout()
for i in range(1, len(city_names)):
    plt.axvline(i - 0.5, color='gray', linestyle=':', linewidth=1)

# Custom legend: colored rectangles for models
handles = [Patch(facecolor=model_palette_50[model], edgecolor='black', label=model) for model in model_names]
ax.legend(handles=handles, title="Model", loc='upper left', bbox_to_anchor=(1,1), fontsize=12, title_fontsize=13)

plt.savefig("dotplot_bias_100yrRL_5day_precip_cities_poster.png", dpi=1000, bbox_inches="tight")
plt.close()