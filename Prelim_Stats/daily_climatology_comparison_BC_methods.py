import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import config

dOTC_path = config.BIAS_CORRECTED_DIR + "/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc"
EQM_path_temp = config.BIAS_CORRECTED_DIR + "/EQM/temp_BC_bicubic_r01.nc"
QDM_path_temp = config.BIAS_CORRECTED_DIR + "/QDM/temp_BC_bicubic_r01.nc"
temp_obs_file = config.TARGET_DIR + "/TabsD_1971_2023.nc"

dOTC_ds = xr.open_dataset(dOTC_path)
EQM_temp_ds = xr.open_dataset(EQM_path_temp)
QDM_temp_ds = xr.open_dataset(QDM_path_temp)
obs_temp_ds = xr.open_dataset(temp_obs_file)

time_slice = slice("1981-01-01", "2010-12-31")

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




tol_colors = ["#4477AA", "#EE7733", "#228833"]  # blue, orange, green

cmap = mcolors.ListedColormap(tol_colors)
labels = ["dOTC+bicubic", "EQM+bicubic", "QDM+bicubic"]

fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True, dpi=200)
im = ax.imshow(winner, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=2)
ax.set_title("Best BC Method for climatology of daily temperature, 1981-2010 (PSS)", fontsize=20)
ax.set_xticks([])
ax.set_yticks([])

legend_elements = [Patch(facecolor=tol_colors[i], label=labels[i]) for i in range(3)]
ax.legend(handles=legend_elements, loc='lower right', fontsize=16, frameon=True)

cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], orientation='vertical', fraction=0.03, pad=0.02)
cbar.ax.set_yticklabels(labels, fontsize=18, fontname="DejaVu Serif")
cbar.set_label("Best Bias Correction Method (PSS)", fontsize=16)
cbar.ax.tick_params(length=0)

plt.savefig("gridwise_pss_winner_temp_climatology_1981_2010.png", dpi=1000)
plt.close()