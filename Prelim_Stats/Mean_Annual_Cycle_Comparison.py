import config
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import argparse


## Code Citations

## License: MIT for part of the code
#https://github.com/jbofill10/Hotel-Booking-Demand-EDA/tree/d91b9fc56693eef20837417f238687704edbce9d/booking_timespans/CityHotelBookingTimeSpan.py

parser = argparse.ArgumentParser(description="City-specific RMSE vs Quantile")
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

def closest_idx(ds, lat, lon):
    lat_arr = ds['lat'].values
    lon_arr = ds['lon'].values
    lat_idx = np.abs(lat_arr - lat).argmin()
    lon_idx = np.abs(lon_arr - lon).argmin()
    return lat_idx, lon_idx

obs_lat_idx, obs_lon_idx = closest_idx(obs_ds, lat, lon)
bicubic_lat_idx, bicubic_lon_idx = closest_idx(bicubic_ds, lat, lon)
coarse_lat_idx, coarse_lon_idx = closest_idx(coarse_ds, lat, lon)
bc_lat_idx, bc_lon_idx = closest_idx(bc_ds, lat, lon)
bc_unet1971_lat_idx, bc_unet1971_lon_idx = closest_idx(bc_unet1971_ds, lat, lon)
bc_unet1771_lat_idx, bc_unet1771_lon_idx = closest_idx(bc_unet1771_ds, lat, lon)

#Only for calibration period
start = "1981-01-01"
end = "2010-12-31"

#Calculating 30 year climatological mean annual cycle
def get_annual_cycle(ds, var, lat_idx, lon_idx):
    data = ds[var].sel(time=slice(start, end)).values[:, lat_idx, lon_idx]
    time = ds['time'].sel(time=slice(start, end)).values
    months = np.array([t.astype('datetime64[M]').astype(int) % 12 + 1 for t in time])
    cycle = np.array([np.nanmean(data[months == m]) for m in range(1, 13)])
    return cycle

annual_cycles = {
    "Observed": get_annual_cycle(obs_ds, "RhiresD", obs_lat_idx, obs_lon_idx),
    "Bicubically interpolated model output": get_annual_cycle(bicubic_ds, "precip", bicubic_lat_idx, bicubic_lon_idx),
    "Coarse model output": get_annual_cycle(coarse_ds, "precip", coarse_lat_idx, coarse_lon_idx),
    "Bias Corrected using EQM": get_annual_cycle(bc_ds, "precip", bc_lat_idx, bc_lon_idx),
    "BC+Downscaled with UNet 1971": get_annual_cycle(bc_unet1971_ds, "precip", bc_unet1971_lat_idx, bc_unet1971_lon_idx),
    "BC+Downscaled with UNet 1771": get_annual_cycle(bc_unet1771_ds, "precip", bc_unet1771_lat_idx, bc_unet1771_lon_idx),
}

plt.figure(figsize=(10, 6))
months = np.arange(1, 13)
for label, cycle in annual_cycles.items():
    plt.plot(months, cycle, marker='o', label=label)
plt.xticks(months, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel("Months")
plt.ylabel("Mean Precip (mm/day)")
plt.title(f"Climatological Mean Annual Cycle (1981-2010) at lat={lat:.3f}, lon={lon:.3f}")
plt.legend()
plt.tight_layout()
plt.savefig(f"{config.OUTPUTS_DIR}/Mean_Annual_Cycle_Comparison_{lat:.3f}_{lon:.3f}.png", dpi=1000)
plt.close()
