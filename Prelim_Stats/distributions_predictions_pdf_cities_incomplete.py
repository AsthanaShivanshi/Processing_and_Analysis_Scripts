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
    """
    Transform Swiss LV95 (E, N) grid arrays to WGS84 (lon, lat) using cs2cs.
    """
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

def plot_city_pdf(city_coords, obs, unet, bicubic, varname, city_name="City"):
    obs_N, obs_E, obs_N_dim, obs_E_dim = get_lat_lon(obs)
    unet_lat, unet_lon, unet_lat_dim, unet_lon_dim = get_lat_lon(unet)
    bicubic_N, bicubic_E, bicubic_N_dim, bicubic_E_dim = get_lat_lon(bicubic)

    # Transform obs grid (E, N) to (lon, lat) 
    obs_E_grid, obs_N_grid = np.meshgrid(obs_E, obs_N)
    obs_lon_grid, obs_lat_grid = swiss_lv95_grid_to_wgs84(obs_E_grid, obs_N_grid)

    # Transform bicubic grid (E, N) to (lon, lat)
    bicubic_E_grid, bicubic_N_grid = np.meshgrid(bicubic_E, bicubic_N)
    bicubic_lon_grid, bicubic_lat_grid = swiss_lv95_grid_to_wgs84(bicubic_E_grid, bicubic_N_grid)

    city_lat, city_lon = city_coords

    dist_obs = np.sqrt((obs_lat_grid - city_lat)**2 + (obs_lon_grid - city_lon)**2)
    lat_idx, lon_idx = np.unravel_index(np.argmin(dist_obs), dist_obs.shape)
    obs_N_val = obs_N[lat_idx]
    obs_E_val = obs_E[lon_idx]
    print(f"[DEBUG] Using obs grid cell lat_idx={lat_idx}, lon_idx={lon_idx}, N={obs_N_val}, E={obs_E_val}")

    obs_series = obs.isel({obs_N_dim: lat_idx, obs_E_dim: lon_idx}).values.flatten()
    print(f"[DEBUG] obs_series: shape={obs_series.shape}, min={np.nanmin(obs_series)}, max={np.nanmax(obs_series)}, nans={np.isnan(obs_series).sum()}")

    unet_series = unet.sel(lat=city_lat, lon=city_lon, method="nearest").values.flatten()
    print(f"[DEBUG] unet_series: shape={unet_series.shape}, min={np.nanmin(unet_series)}, max={np.nanmax(unet_series)}, nans={np.isnan(unet_series).sum()}")

    dist_bicubic = np.sqrt((bicubic_lat_grid - city_lat)**2 + (bicubic_lon_grid - city_lon)**2)
    lat_idx_bicubic, lon_idx_bicubic = np.unravel_index(np.argmin(dist_bicubic), dist_bicubic.shape)
    bicubic_series = bicubic.isel({bicubic_N_dim: lat_idx_bicubic, bicubic_E_dim: lon_idx_bicubic}).values.flatten()
    print(f"[DEBUG] bicubic_series: shape={bicubic_series.shape}, min={np.nanmin(bicubic_series)}, max={np.nanmax(bicubic_series)}, nans={np.isnan(bicubic_series).sum()}")

    obs_series = obs_series[~np.isnan(obs_series)]
    unet_series = unet_series[~np.isnan(unet_series)]
    bicubic_series = bicubic_series[~np.isnan(bicubic_series)]

    print(f"obs_series length: {len(obs_series)}")
    print(f"unet_series length: {len(unet_series)}")
    print(f"bicubic_series length: {len(bicubic_series)}")

    plt.figure(figsize=(8,6))
    plt.hist(obs_series, bins=50, density=True, histtype="step", linewidth=2, color="green", label="Observations from test set 2011-2020")
    plt.hist(unet_series, bins=50, density=True, histtype="step", linewidth=2, color="blue", label="UNet predictions: 2011-2020")
    plt.hist(bicubic_series, bins=50, density=True, histtype="step", linewidth=2, color="orange", label="Bicubic baseline: 2011-2020")
    plt.title(f"{varname} PDF at {city_name} (lat={city_lat:.4f}, lon={city_lon:.4f})")
    plt.xlabel(varname)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../Outputs/pdf_{varname}_{city_name}_latlon_distance_UNet_pred.png")
    plt.close()

if __name__ == "__main__":
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    city_coords = (46.2044, 6.1432)
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
    unet_ds = xr.open_dataset(
        str(BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "models_UNet" / "UNet_Deterministic_Training_Dataset" / "downscaled_predictions_2011_2020_ds.nc"),
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

    print(obs_ds)
    print(unet_ds)
    print(bicubic_ds)
    plot_city_pdf(
        city_coords,
        obs_ds[obs_var],
        unet_ds[obs_var],
        bicubic_ds[obs_var],
        varname=obs_var,
        city_name=city_name
    )