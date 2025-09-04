import config
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

## Code Citations

## License: MIT for part of the code about choosing and grouping into months for mean annual cycle
#https://github.com/jbofill10/Hotel-Booking-Demand-EDA/tree/d91b9fc56693eef20837417f238687704edbce9d/booking_timespans/CityHotelBookingTimeSpan.py

parser = argparse.ArgumentParser(description="Climatology of Mean Annual Cycle")
parser.add_argument("--var", type=int, required=True, help="Variable index (0-3)")
parser.add_argument("--city", type=str, required=True, help="City name")
parser.add_argument("--lat", type=float, required=True, help="Latitude of city")
parser.add_argument("--lon", type=float, required=True, help="Longitude of city")
args = parser.parse_args()

obs_path = f"{config.TARGET_DIR}/TabsD_1971_2023.nc" #Spatial analysis
coarse_path = f"{config.MODELS_DIR}/temp_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/temp_r01_coarse_masked.nc" #Without BC or bicubic, plain RCM run
bc_path = f"{config.BIAS_CORRECTED_DIR}/EQM/temp_QM_BC_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_r01.nc" #BC at coarse resolution (12kms)

bicubic_path = f"{config.BIAS_CORRECTED_DIR}/EQM/temp_BC_bicubic_r01.nc"
bc_unet1971_path = f"{config.BIAS_CORRECTED_DIR}/EQM/DOWNSCALED_TRAINING_QM_BC_temp_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_downscaled_r01.nc"
bc_unet1771_path = f"{config.BIAS_CORRECTED_DIR}/EQM/DOWNSCALED_COMBINED_QM_BC_temp_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_downscaled_r01.nc"

obs_ds = xr.open_dataset(obs_path)
bicubic_ds = xr.open_dataset(bicubic_path)
coarse_ds = xr.open_dataset(coarse_path)
bc_ds = xr.open_dataset(bc_path)
bc_unet1971_ds = xr.open_dataset(bc_unet1971_path)
bc_unet1771_ds = xr.open_dataset(bc_unet1771_path)

lat = args.lat
lon = args.lon


def nearest_grid(ds, lat_target, lon_target):
    lat_vals = ds['lat'].values
    lon_vals = ds['lon'].values
    lat_idx = np.argmin(np.abs(lat_vals - lat_target))
    lon_idx = np.argmin(np.abs(lon_vals - lon_target))
    return lat_idx, lon_idx

obs_lat_idx, obs_lon_idx = nearest_grid(obs_ds, lat, lon)
bicubic_lat_idx, bicubic_lon_idx = nearest_grid(bicubic_ds, lat, lon)
coarse_lat_idx, coarse_lon_idx = nearest_grid(coarse_ds, lat, lon)
bc_lat_idx, bc_lon_idx = nearest_grid(bc_ds, lat, lon)
bc_unet1971_lat_idx, bc_unet1971_lon_idx = nearest_grid(bc_unet1971_ds, lat, lon)
bc_unet1771_lat_idx, bc_unet1771_lon_idx = nearest_grid(bc_unet1771_ds, lat, lon)

#Only for calibration period
start = "1981-01-01"
end = "2010-12-31"


def get_daily_climatology(ds, var, lat, lon):
    N_idx, E_idx = nearest_grid(ds, lat, lon)
    data = ds[var].sel(time=slice(start, end)).isel(N=N_idx, E=E_idx).values
    time = ds['time'].sel(time=slice(start, end)).values
    doy = pd.to_datetime(time).dayofyear
    unique_doy = np.arange(1, 367)  # 1 to 366 (leap years included)
    daily_clim = np.array([np.nanmean(data[doy == d]) for d in unique_doy])
    return daily_clim, unique_doy

annual_cycles = {}
doy_axis = None
for label, ds, var in [
    ("MeteoSwiss Spatial Analysis", obs_ds, "TabsD"),
    ("Coarse non-BC Model O/P", coarse_ds, "temp"),
    ("Bias Corrected using EQM", bc_ds, "temp"),
    ("BC+ Bicubic Model O/P", bicubic_ds, "temp"),
    ("BC+Bicubic+UNet1771 Downscaled", bc_unet1771_ds, "temp"),
    ("BC+Bicubic+UNet1971 Downscaled", bc_unet1971_ds, "temp"),
]:
    clim, doy = get_daily_climatology(ds, var, lat, lon)
    annual_cycles[label] = clim
    if doy_axis is None:
        doy_axis = doy


def perkins_skill_score(obs, model, nbins=12):
    combined = np.concatenate([obs, model])
    bins = np.linspace(np.nanmin(combined), np.nanmax(combined), nbins + 1)
    hist_obs, _ = np.histogram(obs, bins=bins, density=True)
    hist_model, _ = np.histogram(model, bins=bins, density=True)
    hist_obs = hist_obs / np.sum(hist_obs)
    hist_model = hist_model / np.sum(hist_model)
    return np.sum(np.minimum(hist_obs, hist_model))


pss_scores = {}
obs_cycle = annual_cycles["MeteoSwiss Spatial Analysis"]
for label, cycle in annual_cycles.items():
    if label != "MeteoSwiss Spatial Analysis":
        pss = perkins_skill_score(obs_cycle, cycle)
        pss_scores[label] = pss
    else:
        pss_scores[label] = None  # NAN

plt.figure(figsize=(12, 8))

def circular_rolling_mean(series, window):
    pad = window // 2
    padded = np.concatenate([series[-pad:], series, series[:pad]])
    rolled = pd.Series(padded).rolling(window, center=True, min_periods=1).mean().values
    return rolled[pad:-pad]
window = 31

for label, cycle in annual_cycles.items():
    cycle_smooth = circular_rolling_mean(cycle, window)
    if pss_scores[label] is not None:
        legend_label = f"{label} (PSS={pss_scores[label]:.4f})"
    else:
        legend_label = label
    plt.plot(doy_axis, cycle_smooth, label=legend_label)

# ref yr for month ticks
ref_year = 2000  # leap, for Feb 29 cases, otherwise was giving some error
month_starts = pd.date_range(f"{ref_year}-01-01", f"{ref_year}-12-31", freq='MS').dayofyear
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(month_starts, month_labels, fontsize=18, fontname="Times New Roman")
plt.yticks(fontsize=18, fontname="Times New Roman")
plt.xlabel("Month", fontsize=18, fontname="Times New Roman")
plt.ylabel(f"Daily Temperature (°C)", fontsize=18, fontname="Times New Roman")
plt.title(f"Climatology of Temperature (1981-2010) for \n{args.city} (lat={lat:.3f}, lon={lon:.3f})", fontsize=22, fontname="Times New Roman")
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(f"{config.OUTPUTS_DIR}/Temp_Daily_Climatology_Comparison_{args.city}_{lat:.3f}_{lon:.3f}_.png", dpi=1000)
plt.close()