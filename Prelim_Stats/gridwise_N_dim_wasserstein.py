import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import config
from matplotlib.patches import Patch
from scipy.stats import wasserstein_distance_nd

dOTC_path = config.BIAS_CORRECTED_DIR + "/dOTC/precip_temp_tmin_tmax_bicubic_r01.nc"
EQM_path_temp   = config.BIAS_CORRECTED_DIR + "/EQM/temp_BC_bicubic_r01.nc"
EQM_path_precip = config.BIAS_CORRECTED_DIR + "/EQM/precip_BC_bicubic_r01.nc"
EQM_path_tmin   = config.BIAS_CORRECTED_DIR + "/EQM/tmin_BC_bicubic_r01.nc"
EQM_path_tmax   = config.BIAS_CORRECTED_DIR + "/EQM/tmax_BC_bicubic_r01.nc"
QDM_path_temp   = config.BIAS_CORRECTED_DIR + "/QDM/temp_BC_bicubic_r01.nc"
QDM_path_precip = config.BIAS_CORRECTED_DIR + "/QDM/precip_BC_bicubic_r01.nc"
QDM_path_tmin   = config.BIAS_CORRECTED_DIR + "/QDM/tmin_BC_bicubic_r01.nc"
QDM_path_tmax   = config.BIAS_CORRECTED_DIR + "/QDM/tmax_BC_bicubic_r01.nc"
temp_obs_file = config.TARGET_DIR + "/TabsD_1971_2023.nc"
precip_obs_file = config.TARGET_DIR + "/RhiresD_1971_2023.nc"
tmin_obs_file = config.TARGET_DIR + "/TminD_1971_2023.nc"
tmax_obs_file = config.TARGET_DIR + "/TmaxD_1971_2023.nc"

# Load model data for all variables for each BC method
dOTC_ds = xr.open_dataset(dOTC_path)
EQM_temp_ds   = xr.open_dataset(EQM_path_temp)
EQM_precip_ds = xr.open_dataset(EQM_path_precip)
EQM_tmin_ds   = xr.open_dataset(EQM_path_tmin)
EQM_tmax_ds   = xr.open_dataset(EQM_path_tmax)
QDM_temp_ds   = xr.open_dataset(QDM_path_temp)
QDM_precip_ds = xr.open_dataset(QDM_path_precip)
QDM_tmin_ds   = xr.open_dataset(QDM_path_tmin)
QDM_tmax_ds   = xr.open_dataset(QDM_path_tmax)
obs_temp_ds   = xr.open_dataset(temp_obs_file)
obs_precip_ds = xr.open_dataset(precip_obs_file)
obs_tmin_ds = xr.open_dataset(tmin_obs_file)
obs_tmax_ds = xr.open_dataset(tmax_obs_file)




time_slice = slice("1981-01-01", "2010-12-31")

# dOTC
dOTC_temp   = dOTC_ds["temp"].sel(time=time_slice)
dOTC_precip = dOTC_ds["precip"].sel(time=time_slice)
dOTC_tmin   = dOTC_ds["tmin"].sel(time=time_slice)
dOTC_tmax   = dOTC_ds["tmax"].sel(time=time_slice)

# EQM
EQM_temp   = EQM_temp_ds["temp"].sel(time=time_slice)
EQM_precip = EQM_precip_ds["precip"].sel(time=time_slice)
EQM_tmin   = EQM_tmin_ds["tmin"].sel(time=time_slice)
EQM_tmax   = EQM_tmax_ds["tmax"].sel(time=time_slice)

# QDM
QDM_temp   = QDM_temp_ds["temp"].sel(time=time_slice)
QDM_precip = QDM_precip_ds["precip"].sel(time=time_slice)
QDM_tmin   = QDM_tmin_ds["tmin"].sel(time=time_slice)
QDM_tmax   = QDM_tmax_ds["tmax"].sel(time=time_slice)

# Observations
obs_temp    = obs_temp_ds["TabsD"].sel(time=time_slice)
obs_precip  = obs_precip_ds["RhiresD"].sel(time=time_slice)
obs_tmin    = obs_tmin_ds["TminD"].sel(time=time_slice)
obs_tmax    = obs_tmax_ds["TmaxD"].sel(time=time_slice)

def mean_annual_cycle(da):
    return da.groupby('time.dayofyear').mean('time').values

def stack_vars(temp, precip, tmin, tmax):
    return np.stack([temp, precip, tmin, tmax], axis=1)  # (366, 4, lat, lon). ,,,for n dim wasserstein calculation

# MAC
dOTC_cycle = stack_vars(
    mean_annual_cycle(dOTC_temp),
    mean_annual_cycle(dOTC_precip),
    mean_annual_cycle(dOTC_tmin),
    mean_annual_cycle(dOTC_tmax)
)
EQM_cycle = stack_vars(
    mean_annual_cycle(EQM_temp),
    mean_annual_cycle(EQM_precip),
    mean_annual_cycle(EQM_tmin),
    mean_annual_cycle(EQM_tmax)
)
QDM_cycle = stack_vars(
    mean_annual_cycle(QDM_temp),
    mean_annual_cycle(QDM_precip),
    mean_annual_cycle(QDM_tmin),
    mean_annual_cycle(QDM_tmax)
)
obs_cycle = stack_vars(
    mean_annual_cycle(obs_temp),
    mean_annual_cycle(obs_precip),
    mean_annual_cycle(obs_tmin),
    mean_annual_cycle(obs_tmax)
)

# Mask for Swiss domain
tabsd_mask = ~np.isnan(obs_temp.isel(time=0).values)

def gridwise_nd_wasserstein(a, b):
    # a, b: (366, 4, lat, lon)
    wass4D = np.full(a.shape[2:], np.nan)
    for i in range(a.shape[2]):
        for j in range(a.shape[3]):
            a1 = a[:, :, i, j]
            b1 = b[:, :, i, j]
            mask = ~np.any(np.isnan(a1), axis=1) & ~np.any(np.isnan(b1), axis=1)
            if np.sum(mask) > 10:
                try:
                    wass4D[i, j] = wasserstein_distance_nd(a1[mask], b1[mask])
                except Exception:
                    wass4D[i, j] = np.nan
    return wass4D

# Calculate 4D Wasserstein distances
wass_dOTC = gridwise_nd_wasserstein(dOTC_cycle, obs_cycle)
wass_EQM  = gridwise_nd_wasserstein(EQM_cycle, obs_cycle)
wass_QDM  = gridwise_nd_wasserstein(QDM_cycle, obs_cycle)

# Winner:lowest 4D Wasserstein distance
winner = np.full(wass_dOTC.shape, np.nan)
for i in range(wass_dOTC.shape[0]):
    for j in range(wass_dOTC.shape[1]):
        if not tabsd_mask[i, j]:
            continue
        vals = [wass_dOTC[i, j], wass_EQM[i, j], wass_QDM[i, j]]
        if np.any(np.isnan(vals)):
            continue
        winner[i, j] = np.argmin(vals)
winner[~tabsd_mask] = np.nan


cmap = mcolors.ListedColormap(["#0072B2", "#D55E00", "#009E73"])
labels = ["dOTC+bicubic", "EQM+bicubic", "QDM+bicubic"]

fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True, dpi=200)
im = ax.imshow(winner, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=2)
ax.set_title("4D Wasserstein distance winner\nBias correction methods + bicubic, 1981-2010", fontsize=22, fontname="Times New Roman")
ax.set_xticks([])
ax.set_yticks([])

legend_elements = [Patch(facecolor=cmap(i), label=labels[i]) for i in range(3)]
ax.legend(handles=legend_elements, loc='lower right', fontsize=16, frameon=True)

plt.savefig(f"{config.OUTPUTS_DIR}/Spatial/wasserstein4d_winner_1981_2010.png", dpi=1000)
plt.show()