import config
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd


## Code Citations

## License: MIT for part of the code about choosing and grouping into months for mean annual cycle
#https://github.com/jbofill10/Hotel-Booking-Demand-EDA/tree/d91b9fc56693eef20837417f238687704edbce9d/booking_timespans/CityHotelBookingTimeSpan.py

parser = argparse.ArgumentParser(description="WDF Comparison for Calibration period")
parser.add_argument("--var", type=int, required=True, help="Variable index (0-3)")
parser.add_argument("--city", type=str, required=True, help="City name")
parser.add_argument("--lat", type=float, required=True, help="Latitude of city")
parser.add_argument("--lon", type=float, required=True, help="Longitude of city")
args = parser.parse_args()

obs_path = f"{config.TARGET_DIR}/RhiresD_1971_2023.nc"
bicubic_path = f"{config.MODELS_DIR}/precip_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/precip_r01_HR_masked.nc"
coarse_path = f"{config.MODELS_DIR}/precip_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/precip_r01_coarse_masked.nc"
bc_path = f"{config.BIAS_CORRECTED_DIR}/EQM/eqm_precip_r01_allcells.nc"
bc_unet1971_path = f"{config.BIAS_CORRECTED_DIR}/EQM/TRAINING_EQM_precip_downscaled_r01.nc"
bc_unet1771_path = f"{config.BIAS_CORRECTED_DIR}/EQM/COMBINED_EQM_precip_downscaled_r01.nc"

obs_ds = xr.open_dataset(obs_path)
bicubic_ds = xr.open_dataset(bicubic_path)
coarse_ds = xr.open_dataset(coarse_path)
bc_ds = xr.open_dataset(bc_path)
bc_unet1971_ds = xr.open_dataset(bc_unet1971_path)
bc_unet1771_ds = xr.open_dataset(bc_unet1771_path)


lat = args.lat
lon = args.lon

#Nearest grid fucn : because the ds had 2D latlon coords but the nearest function expects 1D in xarray
def nearest_grid(ds, lat_target, lon_target):
    lat2d = ds['lat'].values
    lon2d = ds['lon'].values
    dist = np.sqrt((lat2d - lat_target)**2 + (lon2d - lon_target)**2)
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    return idx  # returns (N_idx, E_idx)

obs_lat_idx, obs_lon_idx = nearest_grid(obs_ds, lat, lon)
bicubic_lat_idx, bicubic_lon_idx = nearest_grid(bicubic_ds, lat, lon)
coarse_lat_idx, coarse_lon_idx = nearest_grid(coarse_ds, lat, lon)
bc_lat_idx, bc_lon_idx = nearest_grid(bc_ds, lat, lon)
bc_unet1971_lat_idx, bc_unet1971_lon_idx = nearest_grid(bc_unet1971_ds, lat, lon)
bc_unet1771_lat_idx, bc_unet1771_lon_idx = nearest_grid(bc_unet1771_ds, lat, lon)

#Only for calibration period
start = "1981-01-01"
end = "2010-12-31"

def monthly_wdf(ds, var, lat, lon, threshold=0.1):
    N_idx, E_idx = nearest_grid(ds, lat, lon)
    data = ds[var].sel(time=slice(start, end)).isel(N=N_idx, E=E_idx).values
    time = pd.to_datetime(ds['time'].sel(time=slice(start, end)).values)
    months = time.month
    years = time.year
    wdf = []
    for m in range(1, 13):
        vals = data[months == m]
        wet_days = np.sum(vals > threshold)
        total_days = len(vals)
        wdf.append(wet_days / total_days if total_days > 0 else np.nan)
    return np.array(wdf), np.arange(1, 13)

annual_cycles = {}
month_axis = None
for label, ds, var in [
    ("MeteoSwiss Spatial Analysis", obs_ds, "RhiresD"),
    ("Coarse Model O/P", coarse_ds, "precip"),
    ("Bicubically Interpolated Model O/P", bicubic_ds, "precip"),
    ("Bias Corrected using EQM", bc_ds, "precip"),
    ("BC+UNet1971 Downscaled", bc_unet1971_ds, "precip"),
    ("BC+UNet1771 Downscaled", bc_unet1771_ds, "precip"),
]:
    clim, months = monthly_wdf(ds, var, lat, lon, threshold=0.1)
    annual_cycles[label] = clim
    if month_axis is None:
        month_axis = months


def PSS(obs, model, nbins=12):
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
        pss = PSS(obs_cycle, cycle)
        pss_scores[label] = pss
    else:
        pss_scores[label] = None  # NAN

plt.figure(figsize=(10, 6))
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for label, cycle in annual_cycles.items():
    if pss_scores[label] is not None:
        legend_label = f"{label} (PSS={pss_scores[label]:.4f})"
    else:
        legend_label = label
    plt.plot(month_axis, cycle, marker='o', label=legend_label)

plt.xticks(month_axis, month_labels)
plt.xlabel("Month")
plt.ylabel(f"Monthly Wet Day Frequency for {args.city}")
plt.title(f"Monthly Wet Day Frequency (>0.1 mm) climatological cycle (1981-2010) for {args.city} lat={lat:.3f}, lon={lon:.3f}")
plt.legend()
plt.tight_layout()
plt.savefig(f"{config.OUTPUTS_DIR}/Precip_Monthly_WDF_Comparison_{args.city}_{lat:.3f}_{lon:.3f}_.png", dpi=1000)
plt.close()