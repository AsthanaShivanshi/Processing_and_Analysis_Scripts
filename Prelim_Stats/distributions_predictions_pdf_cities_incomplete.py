import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg') #To avoid GUI issues
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

def plot_city_pdf(city_coords, obs, unet, recon, varname, city_name="City"):
    obs_lat, obs_lon, obs_lat_dim, obs_lon_dim = get_lat_lon(obs)
    unet_lat, unet_lon, unet_lat_dim, unet_lon_dim = get_lat_lon(unet)
    recon_lat, recon_lon, recon_lat_dim, recon_lon_dim = get_lat_lon(recon)

    # Find closest pixel in each grid
    def closest_idx(lat, lon, city_coords):
        lon2d, lat2d = np.meshgrid(lon, lat)
        dist = np.sqrt((lat2d - city_coords[0])**2 + (lon2d - city_coords[1])**2)
        return np.unravel_index(np.argmin(dist), dist.shape)

    # Observational time series
    lat_idx, lon_idx = closest_idx(obs_lat, obs_lon, city_coords)
    obs_series = obs.isel({obs_lat_dim: lat_idx, obs_lon_dim: lon_idx}).values.flatten()

    # UNet time series
    lat_idx_unet, lon_idx_unet = closest_idx(unet_lat, unet_lon, city_coords)
    unet_series = unet.isel({unet_lat_dim: lat_idx_unet, unet_lon_dim: lon_idx_unet}).values.flatten()

    # Reconstructed time series
    lat_idx_recon, lon_idx_recon = closest_idx(recon_lat, recon_lon, city_coords)
    recon_series = recon.isel({recon_lat_dim: lat_idx_recon, recon_lon_dim: lon_idx_recon}).values.flatten()

    # Remove NaNs
    obs_series = obs_series[~np.isnan(obs_series)]
    unet_series = unet_series[~np.isnan(unet_series)]
    recon_series = recon_series[~np.isnan(recon_series)]

    plt.figure(figsize=(8,6))
    plt.hist(obs_series, bins=50, density=True, histtype="step", linewidth=2, label="Observations from test set 2011-2020")
    plt.hist(unet_series, bins=50, density=True, histtype="step", linewidth=2, label="UNet predictions: trained/validated on 1971-2010")
    plt.hist(recon_series, bins=50, density=True, histtype="step", linewidth=2, label="Reconstructed 2011-2020")
    plt.title(f"{varname} PDF at {city_name} ({city_coords[0]:.4f}, {city_coords[1]:.4f})")
    plt.xlabel(varname)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../Outputs/pdf_{varname}_{city_name}_without_long_timeseries_UNet_pred.png")
    plt.close()

if __name__ == "__main__":
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    city_coords = (47.3769, 8.5417)
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
    chunks={"time": 100}
).sel(time=slice("2011-01-01", "2020-12-31"))
    unet_ds = xr.open_dataset(str(BASE_DIR/"sasthana"/"Downscaling"/"Downscaling_Models"/"models_UNet"/"UNet_Deterministic_Training_Dataset"/"downscaled_predictions_2011_2020_ds.nc"), chunks={"time": 100})
    recon_ds = xr.open_dataset(str(BASE_DIR / "raw_data" / "Reconstruction_UniBern_1763_2020" / f"{recon_var}_1763_2020.nc"), chunks={"time": 100}).sel(time=slice("2011-01-01", "2020-12-31"))

    plot_city_pdf(
        city_coords,
        obs_ds[obs_var],
        unet_ds[obs_var],
        recon_ds[recon_var],
        varname=obs_var,
        city_name=city_name
    )