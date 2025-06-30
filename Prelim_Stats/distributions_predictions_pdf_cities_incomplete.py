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

def plot_city_pdf(city_coords, obs, unet, recon, varname, city_name="City"):
    obs_N, obs_E, obs_N_dim, obs_E_dim = get_lat_lon(obs)
    unet_lat, unet_lon, unet_lat_dim, unet_lon_dim = get_lat_lon(unet)
    recon_N, recon_E, recon_N_dim, recon_E_dim = get_lat_lon(recon)

    # Transform obs grid (E, N) to (lon, lat) 
    obs_E_grid, obs_N_grid = np.meshgrid(obs_E, obs_N)
    obs_lon_grid, obs_lat_grid = swiss_lv95_grid_to_wgs84(obs_E_grid, obs_N_grid)

    # Transform recon grid (E, N) to (lon, lat)
    recon_E_grid, recon_N_grid = np.meshgrid(recon_E, recon_N)
    recon_lon_grid, recon_lat_grid = swiss_lv95_grid_to_wgs84(recon_E_grid, recon_N_grid)

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

    dist_recon = np.sqrt((recon_lat_grid - city_lat)**2 + (recon_lon_grid - city_lon)**2)
    lat_idx_recon, lon_idx_recon = np.unravel_index(np.argmin(dist_recon), dist_recon.shape)
    recon_series = recon.isel({recon_N_dim: lat_idx_recon, recon_E_dim: lon_idx_recon}).values.flatten()
    print(f"[DEBUG] recon_series: shape={recon_series.shape}, min={np.nanmin(recon_series)}, max={np.nanmax(recon_series)}, nans={np.isnan(recon_series).sum()}")

    obs_series = obs_series[~np.isnan(obs_series)]
    unet_series = unet_series[~np.isnan(unet_series)]
    recon_series = recon_series[~np.isnan(recon_series)]

    print(f"obs_series length: {len(obs_series)}")
    print(f"unet_series length: {len(unet_series)}")
    print(f"recon_series length: {len(recon_series)}")

    plt.figure(figsize=(8,6))
    plt.hist(obs_series, bins=50, density=True, histtype="step", linewidth=2, color="green", label="Observations from test set 2011-2020")
    plt.hist(unet_series, bins=50, density=True, histtype="step", linewidth=2, color="blue", label="UNet predictions: 2011-2020")
    plt.hist(recon_series, bins=50, density=True, histtype="step", linewidth=2, color="orange", label="Reconstructed 2011-2020")
    plt.title(f"{varname} PDF at {city_name} (lat={city_lat:.4f}, lon={city_lon:.4f})")
    plt.xlabel(varname)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../Outputs/pdf_{varname}_{city_name}_latlon_distance_UNet_pred.png")
    plt.close()

if __name__ == "__main__":
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    city_coords = (46.2044, 6.1432)  # Geneva
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
    recon_ds = xr.open_dataset(
        str(BASE_DIR / "raw_data" / "Reconstruction_UniBern_1763_2020" / f"{recon_var}_1763_2020.nc"),
        chunks={"time": 100}).sel(time=slice("2011-01-01", "2020-12-31"))

    print(obs_ds)
    print(unet_ds)
    print(recon_ds)
    plot_city_pdf(
        city_coords,
        obs_ds[obs_var],
        unet_ds[obs_var],
        recon_ds[recon_var],
        varname=obs_var,
        city_name=city_name
    )