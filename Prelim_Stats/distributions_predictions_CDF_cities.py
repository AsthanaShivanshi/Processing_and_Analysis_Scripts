import os
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

BASE_DIR = Path(os.environ.get("BASE_DIR", "/work/FAC/FGSE/IDYST/tbeucler/downscaling/"))

# Use the same variable mapping logic as in delta_pdf.py
VAR_MAP = {
    "precip": {"hr": "RhiresD", "model": "precip"},
    "temp":   {"hr": "TabsD",   "model": "temp"},
    "tmin":   {"hr": "TminD",   "model": "tmin"},
    "tmax":   {"hr": "TmaxD",   "model": "tmax"},
}

def get_lat_lon(da):
    if 'lat' in da.dims and 'lon' in da.dims:
        return da['lat'].values, da['lon'].values, 'lat', 'lon'
    elif 'N' in da.dims and 'E' in da.dims:
        return da['N'].values, da['E'].values, 'N', 'E'
    elif 'E' in da.dims and 'N' in da.dims:
        return da['N'].values, da['E'].values, 'N', 'E'
    else:
        raise ValueError("Unknown dimension names for lat/lon in DataArray.")

def closest_idx(grid_y, grid_x, target_coords):
    grid_y_flat = grid_y.flatten()
    grid_x_flat = grid_x.flatten()
    dist = np.sqrt((grid_y_flat - target_coords[0])**2 + (grid_x_flat - target_coords[1])**2)
    min_idx_flat = np.argmin(dist)
    if grid_y.ndim == 2:
        idx_y, idx_x = np.unravel_index(min_idx_flat, grid_y.shape)
    else:
        idx_y, idx_x = np.unravel_index(min_idx_flat, (len(grid_y), len(grid_x)))
    return idx_y, idx_x

def swiss_lv95_grid_to_wgs84(E_grid, N_grid):
    E_flat = E_grid.flatten()
    N_flat = N_grid.flatten()
    coords = "\n".join(f"{E} {N}" for E, N in zip(E_flat, N_flat))
    result = subprocess.run(
        ['cs2cs', '-f', '%.8f', '+init=epsg:2056', '+to', '+init=epsg:4326'],
        input=coords + "\n",
        text=True,
        capture_output=True
    )
    lon_lat = [line.split() for line in result.stdout.strip().split('\n')]
    lon = np.array([float(ll[0]) for ll in lon_lat]).reshape(E_grid.shape)
    lat = np.array([float(ll[1]) for ll in lon_lat]).reshape(E_grid.shape)
    return lon, lat

def plot_city_cdf(city_coords, obs, unet_1971, unet_1771, bicubic, varname, city_name="City"):
    obs_N, obs_E, obs_N_dim, obs_E_dim = get_lat_lon(obs)
    unet_lat, unet_lon, unet_lat_dim, unet_lon_dim = get_lat_lon(unet_1971)
    bicubic_N, bicubic_E, bicubic_N_dim, bicubic_E_dim = get_lat_lon(bicubic)

    obs_E_grid, obs_N_grid = np.meshgrid(obs_E, obs_N)
    obs_lon_grid, obs_lat_grid = swiss_lv95_grid_to_wgs84(obs_E_grid, obs_N_grid)

    bicubic_E_grid, bicubic_N_grid = np.meshgrid(bicubic_E, bicubic_N)
    bicubic_lon_grid, bicubic_lat_grid = swiss_lv95_grid_to_wgs84(bicubic_E_grid, bicubic_N_grid)

    city_lat, city_lon = city_coords

    dist_obs = np.sqrt((obs_lat_grid - city_lat)**2 + (obs_lon_grid - city_lon)**2)
    lat_idx, lon_idx = np.unravel_index(np.argmin(dist_obs), dist_obs.shape)
    obs_N_val = obs_N[lat_idx]
    obs_E_val = obs_E[lon_idx]

    obs_series = obs.isel({obs_N_dim: lat_idx, obs_E_dim: lon_idx}).values.flatten()
    unet_series_1971 = unet_1971.sel(lat=city_lat, lon=city_lon, method="nearest").values.flatten()
    unet_series_1771 = unet_1771.sel(lat=city_lat, lon=city_lon, method="nearest").values.flatten()
    dist_bicubic = np.sqrt((bicubic_lat_grid - city_lat)**2 + (bicubic_lon_grid - city_lon)**2)
    lat_idx_bicubic, lon_idx_bicubic = np.unravel_index(np.argmin(dist_bicubic), dist_bicubic.shape)
    bicubic_series = bicubic.isel({bicubic_N_dim: lat_idx_bicubic, bicubic_E_dim: lon_idx_bicubic}).values.flatten()

    mask = ~np.isnan(obs_series)
    obs_series = obs_series[mask]
    unet_series_1971 = unet_series_1971[mask] if unet_series_1971.shape == obs_series.shape else unet_series_1971[~np.isnan(unet_series_1971)]
    unet_series_1771 = unet_series_1771[mask] if unet_series_1771.shape == obs_series.shape else unet_series_1771[~np.isnan(unet_series_1771)]
    bicubic_series = bicubic_series[mask] if bicubic_series.shape == obs_series.shape else bicubic_series[~np.isnan(bicubic_series)]

    plt.figure(figsize=(8,6))
    plt.hist(obs_series, bins=50, density=True, cumulative=True,histtype="step", linewidth=1, color="green", label="Observations 2011-2020")
    plt.hist(unet_series_1971, bins=50, density=True, cumulative=True, histtype="step", linewidth=1, color="blue", label="UNet trained on 1971 time series")
    plt.hist(unet_series_1771, bins=50, density=True, cumulative=True, histtype="step", linewidth=1, color="red", label="UNet trained on 1771 time series")
    plt.hist(bicubic_series, bins=50, density=True, cumulative=True, histtype="step", linewidth=1, color="orange", label="Bicubic baseline from 1971 time series for 2011-2020")
    plt.title(f"{varname} CDF at {city_name} (lat={city_lat:.4f}, lon={city_lon:.4f})")
    plt.xlabel(varname)
    plt.ylabel("CDF")
    plt.legend()
    plt.tight_layout()
    output_path = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "Outputs" / f"CDF_{varname}_{city_name}_latlon_distance_UNet_pred.png"
    plt.savefig(str(output_path), dpi=500)
    plt.close()

if __name__ == "__main__":
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    city_coords = (47.3769, 8.5417) # Example: Zürich
    city_name = "Zürich"

    var_keys = list(VAR_MAP.keys())
    var_key = var_keys[idx]
    hr_var = VAR_MAP[var_key]["hr"]
    model_var = VAR_MAP[var_key]["model"]

    obs_ds = xr.open_dataset(
        str(BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "data_1971_2023" / "HR_files_full" / f"{hr_var}_1971_2023.nc"),
        chunks={"time": 100}).sel(time=slice("2011-01-01", "2020-12-31"))
    unet_ds_1971 = xr.open_dataset(
        str(BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "models_UNet" / "UNet_Deterministic_Training_Dataset" / "Training_Dataset_Downscaled_Predictions_2011_2020.nc"),
        chunks={"time": 100})
    unet_ds_1771 = xr.open_dataset(
        str(BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "models_UNet" / "UNet_Deterministic_Pretraining_Dataset" / "Pretraining_Dataset_Downscaled_Predictions_2011_2020.nc"),
        chunks={"time": 100})

    bicubic_paths = {
        "RhiresD": BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "RhiresD_step3_interp.nc",
        "TabsD":   BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "TabsD_step3_interp.nc",
        "TminD":   BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "TminD_step3_interp.nc",
        "TmaxD":   BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "TmaxD_step3_interp.nc",
    }
    bicubic_path = bicubic_paths[hr_var]
    bicubic_ds = xr.open_dataset(str(bicubic_path), chunks={"time": 100}).sel(time=slice("2011-01-01", "2020-12-31"))

    plot_city_cdf(
        city_coords,
        obs_ds[hr_var],
        unet_ds_1971[hr_var],
        unet_ds_1771[model_var],
        bicubic_ds[hr_var],
        varname=hr_var,
        city_name=city_name
    )