import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

BASE_DIR = Path(os.environ.get("BASE_DIR", "/work/FAC/FGSE/IDYST/tbeucler/downscaling/"))

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

def swiss_lv95_grid_to_wgs84(E_grid, N_grid):
    import subprocess
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

def empirical_cdf(series, x_grid):
    return np.array([np.mean(series <= x) for x in x_grid])

def plot_city_bias_cdf(city_coords, obs, unet_1971, unet_1771, bicubic, varname, city_name="City"):
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
    obs_series = obs.isel({obs_N_dim: lat_idx, obs_E_dim: lon_idx}).values.flatten()
    unet_series_1971 = unet_1971.sel(lat=city_lat, lon=city_lon, method="nearest").values.flatten()
    unet_series_1771 = unet_1771.sel(lat=city_lat, lon=city_lon, method="nearest").values.flatten()
    unet_series_combined = unet_combined[model_var].sel(time=slice("2011-01-01", "2020-12-31")).values
    unet_series_combined = unet_series_combined.sel(lat=city_lat, lon=city_lon, method="nearest").values.flatten()

    dist_bicubic = np.sqrt((bicubic_lat_grid - city_lat)**2 + (bicubic_lon_grid - city_lon)**2)
    lat_idx_bicubic, lon_idx_bicubic = np.unravel_index(np.argmin(dist_bicubic), dist_bicubic.shape)
    bicubic_series = bicubic.isel({bicubic_N_dim: lat_idx_bicubic, bicubic_E_dim: lon_idx_bicubic}).values.flatten()

    mask = (
        ~np.isnan(obs_series)
        & ~np.isnan(unet_series_1971)
        & ~np.isnan(unet_series_1771)
        & ~np.isnan(bicubic_series)
        & ~np.isnan(unet_series_combined)
    )
    
    obs_series = obs_series[mask]
    unet_series_1971 = unet_series_1971[mask]
    unet_series_1771 = unet_series_1771[mask]
    bicubic_series = bicubic_series[mask]
    unet_series_combined = unet_series_combined[mask]

    # Calculate bias (prediction - observation) for each model
    bias_unet_1971 = unet_series_1971 - obs_series
    bias_unet_1771 = unet_series_1771 - obs_series
    bias_bicubic = bicubic_series - obs_series
    #bias_combined = unet_series_combined - obs_series

    # Common x grid for all bias CDFs
    all_bias = np.concatenate([bias_unet_1971, bias_unet_1771, bias_bicubic, bias_combined])
    x_grid = np.linspace(np.nanmin(all_bias), np.nanmax(all_bias), 300)

    cdf_unet_1971 = empirical_cdf(bias_unet_1971, x_grid)
    cdf_unet_1771 = empirical_cdf(bias_unet_1771, x_grid)
    cdf_bicubic = empirical_cdf(bias_bicubic, x_grid)
    cdf_combined= empirical_cdf(bias_combined, x_grid)

    plt.figure(figsize=(8,6))
    plt.plot(x_grid, cdf_unet_1971, color="blue", linewidth=2, label="UNet 1971")
    plt.plot(x_grid, cdf_unet_1771, color="red", linewidth=2, label="UNet 1771")
    plt.plot(x_grid, cdf_bicubic, color="orange", linewidth=2, label="Bicubic")
    plt.plot(x_grid, cdf_combined, color="green", linewidth=2, label="UNet Combined")
    plt.axhline(1, color='black', linestyle='--', linewidth=1)
    plt.title(f"CDF of Bias (Prediction - Obs) for {varname} at {city_name}\n(lat={city_lat:.3f}, lon={city_lon:.3f})")
    plt.xlabel("Bias (Prediction - Obs)")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.tight_layout()
    output_path = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "Outputs" / f"CDF_of_bias_{varname}_{city_name}_latlon_distance_UNet_pred.png"
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
    unet_combined = xr.open_dataset(str(BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "models_UNet" / "UNet_Deterministic_Training_Dataset" / "Combined_Dataset_Downscaled_Predictions_2011_2020.nc"),chunks={"time": 100})   

    bicubic_paths = {
        "RhiresD": BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "RhiresD_step3_interp.nc",
        "TabsD":   BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "TabsD_step3_interp.nc",
        "TminD":   BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "TminD_step3_interp.nc",
        "TmaxD":   BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "TmaxD_step3_interp.nc",
    }
    bicubic_path = bicubic_paths[hr_var]
    bicubic_ds = xr.open_dataset(str(bicubic_path), chunks={"time": 100}).sel(time=slice("2011-01-01", "2020-12-31"))

    plot_city_bias_cdf(
        city_coords,
        obs_ds[hr_var],
        unet_ds_1971[hr_var],
        unet_ds_1771[model_var],
        bicubic_ds[hr_var],
        unet_combined[model_var],
        varname=hr_var,
        city_name=city_name)