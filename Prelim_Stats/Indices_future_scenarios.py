import config
import xarray as xr
import numpy as np
import pandas as pd
import argparse


#2085 : means the future scenario period is 2070-2099 period for thirty years , centered around 2085

parser = argparse.ArgumentParser(description="Indices Comparison (2070-2099)")
parser.add_argument("--city", type=str, required=True, help="City name")
parser.add_argument("--lat", type=float, required=True, help="Latitude of city")
parser.add_argument("--lon", type=float, required=True, help="Longitude of city")
args = parser.parse_args()

bicubic_temp_path = f"{config.MODELS_DIR}/temp_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/temp_r01_HR_masked.nc"
bicubic_tmin_path = f"{config.MODELS_DIR}/tmin_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/tmin_r01_HR_masked.nc"
bicubic_tmax_path = f"{config.MODELS_DIR}/tmax_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/tmax_r01_HR_masked.nc"
bicubic_precip_path = f"{config.MODELS_DIR}/precip_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/precip_r01_HR_masked.nc"

coarse_temp_path = f"{config.MODELS_DIR}/temp_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/temp_r01_coarse_masked.nc"
coarse_tmin_path = f"{config.MODELS_DIR}/tmin_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/tmin_r01_coarse_masked.nc"
coarse_tmax_path = f"{config.MODELS_DIR}/tmax_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/tmax_r01_coarse_masked.nc"
coarse_precip_path = f"{config.MODELS_DIR}/precip_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/precip_r01_coarse_masked.nc"

bc_temp_path = f"{config.BIAS_CORRECTED_DIR}/EQM/eqm_temp_r01_allcells.nc"
bc_tmin_path = f"{config.BIAS_CORRECTED_DIR}/EQM/eqm_tmin_r01_allcells.nc"
bc_tmax_path = f"{config.BIAS_CORRECTED_DIR}/EQM/eqm_tmax_r01_allcells.nc"
bc_precip_path = f"{config.BIAS_CORRECTED_DIR}/EQM/eqm_precip_r01_allcells.nc"

bc_unet1971_temp_path = f"{config.BIAS_CORRECTED_DIR}/EQM/TRAINING_EQM_temp_downscaled_r01.nc"
bc_unet1971_tmin_path = f"{config.BIAS_CORRECTED_DIR}/EQM/TRAINING_EQM_tmin_downscaled_r01.nc"
bc_unet1971_tmax_path = f"{config.BIAS_CORRECTED_DIR}/EQM/TRAINING_EQM_tmax_downscaled_r01.nc"
bc_unet1971_precip_path = f"{config.BIAS_CORRECTED_DIR}/EQM/TRAINING_EQM_precip_downscaled_r01.nc"

bc_unet1771_temp_path = f"{config.BIAS_CORRECTED_DIR}/EQM/COMBINED_EQM_temp_downscaled_r01.nc"
bc_unet1771_tmin_path = f"{config.BIAS_CORRECTED_DIR}/EQM/COMBINED_EQM_tmin_downscaled_r01.nc"
bc_unet1771_tmax_path = f"{config.BIAS_CORRECTED_DIR}/EQM/COMBINED_EQM_tmax_downscaled_r01.nc"
bc_unet1771_precip_path = f"{config.BIAS_CORRECTED_DIR}/EQM/COMBINED_EQM_precip_downscaled_r01.nc"

datasets = {
    "Coarse Model O/P": {
        "temp": (coarse_temp_path, "temp"),
        "tmin": (coarse_tmin_path, "tmin"),
        "tmax": (coarse_tmax_path, "tmax"),
        "precip": (coarse_precip_path, "precip"),
    },
    "Bicubically Interpolated Model O/P": {
        "temp": (bicubic_temp_path, "temp"),
        "tmin": (bicubic_tmin_path, "tmin"),
        "tmax": (bicubic_tmax_path, "tmax"),
        "precip": (bicubic_precip_path, "precip"),
    },
    "Bias Corrected using EQM": {
        "temp": (bc_temp_path, "temp"),
        "tmin": (bc_tmin_path, "tmin"),
        "tmax": (bc_tmax_path, "tmax"),
        "precip": (bc_precip_path, "precip"),
    },
    "BC+UNet1971 Downscaled": {
        "temp": (bc_unet1971_temp_path, "temp"),
        "tmin": (bc_unet1971_tmin_path, "tmin"),
        "tmax": (bc_unet1971_tmax_path, "tmax"),
        "precip": (bc_unet1971_precip_path, "precip"),
    },
    "BC+UNet1771 Downscaled": {
        "temp": (bc_unet1771_temp_path, "temp"),
        "tmin": (bc_unet1771_tmin_path, "tmin"),
        "tmax": (bc_unet1771_tmax_path, "tmax"),
        "precip": (bc_unet1771_precip_path, "precip"),
    },
}

lat = args.lat
lon = args.lon
start = "2070-01-01"
end = "2099-12-31"

#Centred around 2085

def nearest_grid(ds, lat_target, lon_target):
    lat2d = ds['lat'].values
    lon2d = ds['lon'].values
    dist = np.sqrt((lat2d - lat_target)**2 + (lon2d - lon_target)**2)
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    return idx  # (N_idx, E_idx)

def get_series(ds, var, lat, lon):
    N_idx, E_idx = nearest_grid(ds, lat, lon)
    data = ds[var].sel(time=slice(start, end)).isel(N=N_idx, E=E_idx).values
    time = pd.to_datetime(ds['time'].sel(time=slice(start, end)).values)
    return data, time

results = {}

for label, paths in datasets.items():
    ds_temp = xr.open_dataset(paths["temp"][0])
    ds_tmin = xr.open_dataset(paths["tmin"][0])
    ds_tmax = xr.open_dataset(paths["tmax"][0])
    ds_precip = xr.open_dataset(paths["precip"][0])

    temp, time = get_series(ds_temp, paths["temp"][1], lat, lon)
    tmin, _ = get_series(ds_tmin, paths["tmin"][1], lat, lon)
    tmax, _ = get_series(ds_tmax, paths["tmax"][1], lat, lon)
    precip, _ = get_series(ds_precip, paths["precip"][1], lat, lon)

    # Summer mask for JJA
    summer_mask = (time.month >= 6) & (time.month <= 8)

    # TN 
    tropical_nights_bool = tmin[summer_mask] > 20
    tropical_nights = np.mean(tropical_nights_bool)
    tropical_nights_95 = np.percentile(tropical_nights_bool.astype(float), 95)

    # HD
    hot_days_bool = tmax[summer_mask] > 30
    hot_days = np.mean(hot_days_bool)
    hot_days_95 = np.percentile(hot_days_bool.astype(float), 95)

    # DTR: min, max, and 95th pctl
    dtr_series = tmax - tmin
    dtr_min = np.min(dtr_series)
    dtr_max = np.max(dtr_series)
    dtr_95 = np.percentile(dtr_series, 95)

    # Rolling window of 5 days, max sum() and 95th percentile over future period
    precip_5day_series = pd.Series(precip).rolling(window=5).sum().dropna().values
    precip_5day_max = np.max(precip_5day_series)
    precip_5day_95 = np.percentile(precip_5day_series, 95)

    results[label] = [
        tropical_nights_95,
        hot_days_95,
        dtr_min,
        dtr_max,
        dtr_95,
        precip_5day_max,
        precip_5day_95
    ]

indices = [
    "Tropical Nights (JJA, Tmin>20°C) 95th Percentile",
    "Hot Days (JJA, Tmax>30°C) 95th Percentile",
    "Min Diurnal Temperature Range (°C) over Future Period",
    "Max Diurnal Temperature Range (°C) over Future Period",
    "95th Percentile Diurnal Temperature Range (°C)",
    "Max Consecutive 5-Day Precipitation (mm) over Future Period",
    "95th Percentile Consecutive 5-Day Precipitation (mm)"
]
df = pd.DataFrame(results, index=indices)

csv_path = f"{config.OUTPUTS_DIR}/FUTURE_Indices_Comparison_{args.city}_{lat:.3f}_{lon:.3f}_2070_2099.csv"
df.to_csv(csv_path)
print(f"Saved indices to {csv_path}")