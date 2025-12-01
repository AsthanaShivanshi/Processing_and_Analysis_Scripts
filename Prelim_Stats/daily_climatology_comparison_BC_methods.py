import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import config
import seaborn as sns
sns.set(style="whitegrid")
from matplotlib.colors import ListedColormap
from closest_grid_cell import select_nearest_grid_cell

import proplot as pplt


np.Inf=np.inf

dOTC_path = config.BIAS_CORRECTED_DIR + "/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc"
EQM_path_temp = config.BIAS_CORRECTED_DIR + "/EQM/temp_BC_bicubic_r01.nc"
#QDM_path_temp = config.BIAS_CORRECTED_DIR + "/QDM/temp_BC_bicubic_r01.nc"
temp_obs_file = config.TARGET_DIR + "/TabsD_1971_2023.nc"

dOTC_ds = xr.open_dataset(dOTC_path)
EQM_temp_ds = xr.open_dataset(EQM_path_temp)
#QDM_temp_ds = xr.open_dataset(QDM_path_temp)
obs_temp_ds = xr.open_dataset(temp_obs_file)


time_slice = slice("2011-01-01", "2023-12-31")

dOTC_temp = dOTC_ds["temp"].sel(time=time_slice)
EQM_temp = EQM_temp_ds["temp"].sel(time=time_slice)
#QDM_temp = QDM_temp_ds["temp"].sel(time=time_slice)
obs_temp = obs_temp_ds["TabsD"].sel(time=time_slice)

# Mask for Swiss domain
tabsd_mask = ~np.isnan(obs_temp.isel(time=0).values)


def daily_climatology(arr, time_coord):
    dayofyear = time_coord.dt.dayofyear
    clim = np.full((366, arr.shape[1], arr.shape[2]), np.nan)
    for doy in range(1, 367):
        mask = (dayofyear == doy)
        if np.any(mask):
            clim[doy-1] = np.nanmean(arr[mask], axis=0)
        else:
            clim[doy-1] = np.nan  # Explicitly set to nan if no data
    return clim



dOTC_clim = daily_climatology(dOTC_temp.values, dOTC_temp["time"])
EQM_clim  = daily_climatology(EQM_temp.values, EQM_temp["time"])
#QDM_clim  = daily_climatology(QDM_temp.values, QDM_temp["time"])
obs_clim  = daily_climatology(obs_temp.values, obs_temp["time"])



def gridwise_perkins_skill_score(a, b, nbins=50):
    # a, b: (366, lat, lon) daily climatology
    pss = np.full(a.shape[1:], np.nan)
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            a1 = a[:, i, j]
            b1 = b[:, i, j]
            mask = ~np.isnan(a1) & ~np.isnan(b1)
            if np.sum(mask) > 10:
                try:
                    a_valid = a1[mask]
                    b_valid = b1[mask]
                    combined_data = np.concatenate([a_valid, b_valid])
                    bins = np.linspace(np.min(combined_data), np.max(combined_data), nbins + 1)
                    hist_a, _ = np.histogram(a_valid, bins=bins, density=True)
                    hist_b, _ = np.histogram(b_valid, bins=bins, density=True)
                    hist_a = hist_a / np.sum(hist_a)
                    hist_b = hist_b / np.sum(hist_b)
                    pss[i, j] = np.sum(np.minimum(hist_a, hist_b))
                except Exception:
                    pss[i, j] = np.nan
    return pss


# Calculate gridwise PSS for daily climatology
pss_dOTC = gridwise_perkins_skill_score(dOTC_clim, obs_clim)
pss_EQM  = gridwise_perkins_skill_score(EQM_clim, obs_clim)
#pss_QDM  = gridwise_perkins_skill_score(QDM_clim, obs_clim)

# Find winner (highest PSS) at each grid cell
winner = np.full(pss_dOTC.shape, np.nan)
for i in range(pss_dOTC.shape[0]):
    for j in range(pss_dOTC.shape[1]):
        if not tabsd_mask[i, j]:
            continue
        vals = [pss_dOTC[i, j], pss_EQM[i, j]]
        if np.any(np.isnan(vals)):
            continue
        winner[i, j] = np.argmax(vals)
winner[~tabsd_mask] = np.nan


# Calculate percentages for legend


total_cells = np.sum(tabsd_mask)
percentages = []
for i in range(2):
    count = np.sum((winner == i) & tabsd_mask)
    perc = 100 * count / total_cells
    percentages.append(perc)

labels = [
    f"dOTC+bicubic ({percentages[0]:.1f}% of grid cells)",
    f"EQM+bicubic ({percentages[1]:.1f}% of grid cells)",
    #f"QDM+bicubic ({percentages[2]:.1f}%)"
]

# Print best BC method for each city
cities = {
    "Bern": (46.9480, 7.4474),
    "Geneva": (46.2044, 6.1432),
    "Locarno": (46.1709, 8.7995),
    "Lugano": (46.0037, 8.9511),
    "Zürich": (47.3769, 8.5417),
    "Davos": (46.8027, 9.8360), 
    "St. Moritz": (46.4908, 9.8355), 

    
}

for city, (lat, lon) in cities.items():
    result = select_nearest_grid_cell(obs_temp_ds, lat, lon)
    i, j = result['lat_idx'], result['lon_idx']
    if tabsd_mask[i, j] and not np.isnan(winner[i, j]):
        method_idx = int(winner[i, j])
        method_name = ["dOTC+bicubic interpolation", "EQM+bicubic interpolation" ][method_idx]
        print(f"{city}: Best BC method is {method_name}")
    else:
        print(f"{city}: No valid data at nearest grid cell.")



cbf_colors = ['yellow', 'blue']  # dOTC = yellow, EQM = blue
cmap = ListedColormap(cbf_colors)

fig, ax = plt.subplots(figsize=(14, 12), dpi=1000)

im = ax.imshow(winner, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("", fontsize=32)
ax.set_ylabel("", fontsize=32)

for spine in ax.spines.values():
    spine.set_visible(False)


legend_elements = [
    Patch(facecolor='yellow', label=labels[0]),  # dOTC+bicubic
    Patch(facecolor='blue', label=labels[1])     # EQM+bicubic
]
ax.legend(
    handles=legend_elements,
    loc='upper right',
    bbox_to_anchor=(1.6, 1),  # Offset legend further right
    fontsize=38,
    frameon=False
)


city_markers = {
    "Geneva": (46.2044, 6.1432), 
    "Locarno": (46.1709, 8.7995), 
}

offsets = {
    "Geneva": (30, -5),
    "Locarno": (-3, 3),
}

for city, (lat, lon) in city_markers.items():
    result = select_nearest_grid_cell(obs_temp_ds, lat, lon)
    i, j = result['lat_idx'], result['lon_idx']
    ax.plot(j, i, marker='*', color='gold', markeredgecolor='black', markersize=35, markeredgewidth=3, zorder=10)
    dx, dy = offsets.get(city, (3, -3))
    ax.text(j + dx, i + dy, city, fontsize=32, color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'), zorder=11)

ax.set_title(
    "\"Best\" skill (EQM or dOTC) for Mean Annual Cycle of Daily Temperature\n"
    "(Calibration: 1981-2010, Validation: 2011-2023)",
    fontsize=44, fontweight='bold'
)
plt.savefig("gridwise_pss_winner_temp_climatology_2011_2023_poster.png", dpi=1000, bbox_inches='tight')
plt.close()
