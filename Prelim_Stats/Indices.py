import config
import xarray as xr
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Indices Comparison for Calib Period (1981-2010)")
parser.add_argument("--city", type=str, required=True, help="City name")
parser.add_argument("--lat", type=float, required=True, help="Latitude of city")
parser.add_argument("--lon", type=float, required=True, help="Longitude of city")
args = parser.parse_args()

obs_temp_path = f"{config.TARGET_DIR}/TabsD_1971_2023.nc"
obs_tmin_path = f"{config.TARGET_DIR}/TminD_1971_2023.nc"
obs_tmax_path = f"{config.TARGET_DIR}/TmaxD_1971_2023.nc"
obs_precip_path = f"{config.TARGET_DIR}/RhiresD_1971_2023.nc"

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
    "MeteoSwiss Spatial Analysis": {
        "temp": (obs_temp_path, "TabsD"),
        "tmin": (obs_tmin_path, "TminD"),
        "tmax": (obs_tmax_path, "TmaxD"),
        "precip": (obs_precip_path, "RhiresD"),
    },
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
start = "1981-01-01"
end = "2010-12-31"

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

    # Summer mask for summer indices
    summer_mask = (time.month >= 6) & (time.month <= 8)

    # TN : Average over summer
    tropical_nights = np.mean(tmin[summer_mask] > 20)

    # HD : Average over summer
    hot_days = np.mean(tmax[summer_mask] > 30)

    # DTR: min and max over the cal period
    dtr_series = tmax - tmin
    dtr_min = np.min(dtr_series)
    dtr_max = np.max(dtr_series)

    # Rolling window of 5 days, max sum() over cal period
    precip_5day = pd.Series(precip).rolling(window=5).sum().max()

    results[label] = [
        tropical_nights,
        hot_days,
        dtr_min, 
        dtr_max,
        precip_5day
    ]

indices = [
    "Tropical Nights (JJA, Tmin>20°C) Cal period Mean",
    "Hot Days (JJA, Tmax>30°C) Cal period Mean",
    "Min Diurnal Temperature Range (°C) over Cal Period",
    "Max Diurnal Temperature Range (°C) over Cal Period",
    "Max Consecutive 5-Day Precipitation (mm) over Cal Period"
]
df = pd.DataFrame(results, index=indices)

csv_path = f"{config.OUTPUTS_DIR}/Indices_Comparison_{args.city}_{lat:.3f}_{lon:.3f}_1981_2010.csv"
df.to_csv(csv_path)
print(f"Saved indices to {csv_path}")