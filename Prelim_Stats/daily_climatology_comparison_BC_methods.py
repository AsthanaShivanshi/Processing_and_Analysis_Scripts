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

dOTC_path = config.BIAS_CORRECTED_DIR + "/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc"
EQM_path_temp = config.BIAS_CORRECTED_DIR + "/EQM/temp_BC_bicubic_r01.nc"
QDM_path_temp = config.BIAS_CORRECTED_DIR + "/QDM/temp_BC_bicubic_r01.nc"
temp_obs_file = config.TARGET_DIR + "/TabsD_1971_2023.nc"

dOTC_ds = xr.open_dataset(dOTC_path)
EQM_temp_ds = xr.open_dataset(EQM_path_temp)
QDM_temp_ds = xr.open_dataset(QDM_path_temp)
obs_temp_ds = xr.open_dataset(temp_obs_file)


time_slice = slice("2011-01-01", "2023-12-31")

dOTC_temp = dOTC_ds["temp"].sel(time=time_slice)
EQM_temp = EQM_temp_ds["temp"].sel(time=time_slice)
QDM_temp = QDM_temp_ds["temp"].sel(time=time_slice)
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
QDM_clim  = daily_climatology(QDM_temp.values, QDM_temp["time"])
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
pss_QDM  = gridwise_perkins_skill_score(QDM_clim, obs_clim)

# Find winner (highest PSS) at each grid cell
winner = np.full(pss_dOTC.shape, np.nan)
for i in range(pss_dOTC.shape[0]):
    for j in range(pss_dOTC.shape[1]):
        if not tabsd_mask[i, j]:
            continue
        vals = [pss_dOTC[i, j], pss_EQM[i, j], pss_QDM[i, j]]
        if np.any(np.isnan(vals)):
            continue
        winner[i, j] = np.argmax(vals)
winner[~tabsd_mask] = np.nan


# Calculate percentages for legend


total_cells = np.sum(tabsd_mask)
percentages = []
for i in range(3):
    count = np.sum((winner == i) & tabsd_mask)
    perc = 100 * count / total_cells
    percentages.append(perc)

labels = [
    f"dOTC+bicubic ({percentages[0]:.1f}%)",
    f"EQM+bicubic ({percentages[1]:.1f}%)",
    f"QDM+bicubic ({percentages[2]:.1f}%)"
]

# Print best BC method for each city
cities = {
    "Bern": (46.9480, 7.4474),
    "Geneva": (46.2044, 6.1432),
    "Locarno": (46.1709, 8.7995),
    "Lugano": (46.0037, 8.9511),
    "Zürich": (47.3769, 8.5417)
}

for city, (lat, lon) in cities.items():
    result = select_nearest_grid_cell(obs_temp_ds, lat, lon)
    i, j = result['lat_idx'], result['lon_idx']
    if tabsd_mask[i, j] and not np.isnan(winner[i, j]):
        method_idx = int(winner[i, j])
        method_name = ["dOTC+bicubic", "EQM+bicubic", "QDM+bicubic"][method_idx]
        print(f"{city}: Best BC method is {method_name}")
    else:
        print(f"{city}: No valid data at nearest grid cell.")


cbf_colors = plt.get_cmap('Set1').colors[:3]  # First 3 colors from Set1
cmap = ListedColormap(cbf_colors)
# labels is now updated above

fig, ax = plt.subplots(figsize=(10, 9), dpi=1000)
im = ax.imshow(winner, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=2)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")

for spine in ax.spines.values():
    spine.set_visible(False)

legend_elements = [Patch(facecolor=cbf_colors[i], label=labels[i]) for i in range(3)]
ax.legend(handles=legend_elements, loc='upper left', fontsize=14, frameon=False, bbox_to_anchor=(1.05, 1))


city_markers = {
    "Zürich": (47.3769, 8.5417),   #EQM best
    "Bern": (46.9480, 7.4474), #dOTC best
    "Locarno": (46.1709, 8.7995) , #QDM best
}


offsets = {
    "Zürich": (3, -3),
    "Bern": (-30, 5),
    "Locarno": (3, 5)
}



for city, (lat, lon) in city_markers.items():
    result = select_nearest_grid_cell(obs_temp_ds, lat, lon)
    i, j = result['lat_idx'], result['lon_idx']
    ax.plot(j, i, marker='*', color='gold', markeredgecolor='black', markersize=18, markeredgewidth=2, zorder=10)
    dx, dy = offsets.get(city, (3, -3))
    ax.text(j + dx, i + dy, city, fontsize=14, color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'), zorder=11)


ax.set_title("Best Bias Correction for Daily Temperature Decadal Mean Annual Cycle \n(Perkins Skill Score, 2011–2023)", fontsize=18, pad=15)
plt.savefig("gridwise_pss_winner_temp_climatology_2011_2023_poster.png", dpi=1000, bbox_inches='tight')
plt.close()

