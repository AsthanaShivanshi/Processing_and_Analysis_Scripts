import os
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

BASE_DIR = Path(os.environ.get("BASE_DIR", "/work/FAC/FGSE/IDYST/tbeucler/downscaling/"))

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

def plot_city_delta_pdf(city_coords, obs, unet_1971, unet_1771, unet_combined, bicubic, varname, city_name="City"):
    obs_N, obs_E, obs_N_dim, obs_E_dim = get_lat_lon(obs)
    unet_lat, unet_lon, unet_lat_dim, unet_lon_dim = get_lat_lon(unet_1971)
    bicubic_N, bicubic_E, bicubic_N_dim, bicubic_E_dim = get_lat_lon(bicubic)

    # Transforming obs grid (E, N) to (lon, lat)
    obs_E_grid, obs_N_grid = np.meshgrid(obs_E, obs_N)
    obs_lon_grid, obs_lat_grid = swiss_lv95_grid_to_wgs84(obs_E_grid, obs_N_grid)

    # Transforming bicubic grid (E, N) to (lon, lat)
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
    unet_series_combined = unet_combined.sel(lat=city_lat, lon=city_lon, method="nearest").values.flatten()
    dist_bicubic = np.sqrt((bicubic_lat_grid - city_lat)**2 + (bicubic_lon_grid - city_lon)**2)
    lat_idx_bicubic, lon_idx_bicubic = np.unravel_index(np.argmin(dist_bicubic), dist_bicubic.shape)
    bicubic_series = bicubic.isel({bicubic_N_dim: lat_idx_bicubic, bicubic_E_dim: lon_idx_bicubic}).values.flatten()

    mask = ~np.isnan(obs_series)
    obs_series = obs_series[mask]
    unet_series_1971 = unet_series_1971[mask] if unet_series_1971.shape == obs_series.shape else unet_series_1971[~np.isnan(unet_series_1971)]
    unet_series_1771 = unet_series_1771[mask] if unet_series_1771.shape == obs_series.shape else unet_series_1771[~np.isnan(unet_series_1771)]
    unet_series_combined = unet_series_combined[mask] if unet_series_combined.shape == obs_series.shape else unet_series_combined[~np.isnan(unet_series_combined)]
    bicubic_series = bicubic_series[mask] if bicubic_series.shape == obs_series.shape else bicubic_series[~np.isnan(bicubic_series)]

    # Computing deltas, model -observations
    delta_bicubic = bicubic_series - obs_series
    delta_unet_1971 = unet_series_1971 - obs_series
    delta_unet_1771 = unet_series_1771 - obs_series
    delta_unet_combined = unet_series_combined - obs_series

    plt.figure(figsize=(8,6))
    plt.hist(delta_bicubic, bins=50, density=True, histtype="step", linewidth=2, color="orange", label="Bicubic - Obs")
    plt.hist(delta_unet_1971, bins=50, density=True, histtype="step", linewidth=2, color="blue", label="UNet 1971-2020 - Obs")
    plt.hist(delta_unet_1771, bins=50, density=True, histtype="step", linewidth=2, color="red", label="UNet 1771-2020 - Obs")
    plt.hist(delta_unet_combined, bins=50, density=True, histtype="step", linewidth=2, color="purple", label="UNet Combined - Obs")
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.title(f"{varname} ΔPDF at {city_name} (model - obs, 2011-2020)")
    plt.xlabel(f"{varname} (Model - Obs)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../Outputs/deltaPDF_{varname}_{city_name}_latlon_distance_UNet_pred.png")
    plt.close()

if __name__ == "__main__":
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    city_coords = (46.2044, 6.1432) #To be changed depending on the city 
    city_name = "Geneva"

    varnames = [
        ("RhiresD", "precip"),
        ("TabsD", "temp"),
        ("TminD", "tmin"),
        ("TmaxD", "tmax"),
    ]

    obs_var, recon_var = varnames[idx]

    obs_ds = xr.open_dataset(
        str(BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "data_1971_2023" / "HR_files_full" / f"{obs_var}_1971_2023.nc"),
        chunks={"time": 100}).sel(time=slice("2011-01-01", "2020-12-31"))
    unet_ds_1971 = xr.open_dataset(
        str(BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "models_UNet" / "UNet_Deterministic_Training_Dataset" / "downscaled_predictions_2011_2020_ds.nc"),
        chunks={"time": 100})
    unet_ds_1771 = xr.open_dataset(
        "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/models_UNet/UNet_Deterministic_Pretraining_Dataset/Pretraining_Dataset_Downscaled_Predictions_2011_2020.nc",
        chunks={"time": 100})
    unet_ds_combined = xr.open_dataset(
        "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/models_UNet/UNet_Deterministic_Combined_Dataset/Combined_Dataset_Downscaled_Predictions_2011_2020.nc",
        chunks={"time": 100})

    # Bicubic baseline path mapping
    bicubic_paths = {
        "RhiresD": BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "RhiresD_step3_interp.nc",
        "TabsD":   BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "TabsD_step3_interp.nc",
        "TminD":   BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "TminD_step3_interp.nc",
        "TmaxD":   BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "TmaxD_step3_interp.nc",
    }
    bicubic_path = bicubic_paths[obs_var]
    bicubic_ds = xr.open_dataset(str(bicubic_path), chunks={"time": 100}).sel(time=slice("2011-01-01", "2020-12-31"))

    plot_city_delta_pdf(
        city_coords,
        obs_ds[obs_var],
        unet_ds_1971[obs_var],
        unet_ds_1771[obs_var],
        unet_ds_combined[obs_var],
        bicubic_ds[obs_var],
        varname=obs_var,
        city_name=city_name
    )