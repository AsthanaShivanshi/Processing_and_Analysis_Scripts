import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg') #To avoid GUI issues
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from pyproj import Transformer

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
    """
    Find the indices in (grid_y, grid_x) closest to target_coords = (target_y, target_x).
    grid_y, grid_x: 1D or 2D arrays of coordinates (e.g., N, E or lat, lon)
    target_coords: tuple (target_y, target_x)
    Returns: (idx_y, idx_x)
    """
    # If grid is 2D, flatten for distance calculation
    grid_y_flat = grid_y.flatten()
    grid_x_flat = grid_x.flatten()
    dist = np.sqrt((grid_y_flat - target_coords[0])**2 + (grid_x_flat - target_coords[1])**2)
    min_idx_flat = np.argmin(dist)
    # Convert flat index back to 2D indices
    if grid_y.ndim == 2:
        idx_y, idx_x = np.unravel_index(min_idx_flat, grid_y.shape)
    else:
        idx_y, idx_x = np.unravel_index(min_idx_flat, (len(grid_y), len(grid_x)))
    return idx_y, idx_x

def plot_city_pdf(city_coords, obs, unet, recon, varname, city_name="City"):
    obs_N, obs_E, obs_N_dim, obs_E_dim = get_lat_lon(obs)
    unet_lat, unet_lon, unet_lat_dim, unet_lon_dim = get_lat_lon(unet)
    recon_N, recon_E, recon_N_dim, recon_E_dim = get_lat_lon(recon)

    # Convert city_coords (lat, lon) to Swiss LV95 (E, N)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    city_lat, city_lon = city_coords  # (lat, lon)
    city_E, city_N = transformer.transform(city_lon, city_lat)
    swiss_city_coords = (city_N, city_E)

    # Find closest cell in obs grid (Swiss LV95)
    lat_idx, lon_idx = closest_idx(obs_N, obs_E, swiss_city_coords)
    obs_N_val = obs_N[lat_idx]
    obs_E_val = obs_E[lon_idx]
    print(f"[DEBUG] Using obs grid cell lat_idx={lat_idx}, lon_idx={lon_idx}, N={obs_N_val}, E={obs_E_val}")

    obs_series = obs.isel({obs_N_dim: lat_idx, obs_E_dim: lon_idx}).values.flatten()
    print(f"[DEBUG] obs_series: shape={obs_series.shape}, min={np.nanmin(obs_series)}, max={np.nanmax(obs_series)}, nans={np.isnan(obs_series).sum()}")

    # For UNet, use city_coords (lat, lon)
    lat_idx_unet, lon_idx_unet = closest_idx(unet_lat, unet_lon, city_coords)
    unet_series = unet.isel({unet_lat_dim: lat_idx_unet, unet_lon_dim: lon_idx_unet}).values.flatten()
    print(f"[DEBUG] unet_series: shape={unet_series.shape}, min={np.nanmin(unet_series)}, max={np.nanmax(unet_series)}, nans={np.isnan(unet_series).sum()}")

    # For recon, use Swiss LV95
    lat_idx_recon, lon_idx_recon = closest_idx(recon_N, recon_E, swiss_city_coords)
    recon_series = recon.isel({recon_N_dim: lat_idx_recon, recon_E_dim: lon_idx_recon}).values.flatten()
    print(f"[DEBUG] recon_series: shape={recon_series.shape}, min={np.nanmin(recon_series)}, max={np.nanmax(recon_series)}, nans={np.isnan(recon_series).sum()}")

    # Removing NaNs
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
    plt.title(f"{varname} PDF at {city_name} (N={obs_N_val:.2f}, E={obs_E_val:.2f})")
    plt.xlabel(varname)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../Outputs/pdf_{varname}_{city_name}_without_long_timeseries_UNet_pred.png")
    plt.close()

if __name__ == "__main__":
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    city_coords = (47.3769, 8.5417)  # (lat, lon)
    city_name = "Zürich"

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
        city_name=city_name)