import os
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from scipy.stats import cramervonmises_2samp

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

def empirical_cdf(series, x_grid): #CDF empirical
    return np.array([np.mean(series <= x) for x in x_grid])

def perkins(cdf1, cdf2):  #Sensitive to the shape of the CDFs
    # Perkins SS: calculated as the mean (min) of two CDFs
    return np.mean(np.minimum(cdf1, cdf2))


#CVM test : does not assume any distribtuion, not senstive to mean or variance 

def plot_city_cdf_and_scores(city_coords, obs, unet_1971, unet_1771, bicubic, unet_combined, varname, city_name="City"):
    obs_N, obs_E, obs_N_dim, obs_E_dim = get_lat_lon(obs)
    unet_lat, unet_lon, unet_lat_dim, unet_lon_dim = get_lat_lon(unet_1971)
    bicubic_N, bicubic_E, bicubic_N_dim, bicubic_E_dim = get_lat_lon(bicubic)
    unet_combined_N, unet_combined_E, unet_combined_N_dim, unet_combined_E_dim = get_lat_lon(unet_combined)

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
    dist_bicubic = np.sqrt((bicubic_lat_grid - city_lat)**2 + (bicubic_lon_grid - city_lon)**2)
    lat_idx_bicubic, lon_idx_bicubic = np.unravel_index(np.argmin(dist_bicubic), dist_bicubic.shape)
    bicubic_series = bicubic.isel({bicubic_N_dim: lat_idx_bicubic, bicubic_E_dim: lon_idx_bicubic}).values.flatten()
    unet_combined_N, unet_combined_E, unet_combined_N_dim, unet_combined_E_dim = get_lat_lon(unet_combined[model_var])
    unet_combined_E_grid, unet_combined_N_grid = np.meshgrid(unet_combined_E, unet_combined_N)
    unet_combined_lon_grid, unet_combined_lat_grid = swiss_lv95_grid_to_wgs84(unet_combined_E_grid, unet_combined_N_grid)
    dist_combined = np.sqrt((unet_combined_lat_grid - city_lat)**2 + (unet_combined_lon_grid - city_lon)**2)
    lat_idx_combined, lon_idx_combined = np.unravel_index(np.argmin(dist_combined), dist_combined.shape)
    unet_series_combined = (
        unet_combined[model_var]
        .isel({unet_combined_N_dim: lat_idx_combined, unet_combined_E_dim: lon_idx_combined})
        .sel(time=slice("2011-01-01", "2020-12-31"))
        .values.flatten()
)
    mask = ~np.isnan(obs_series) & ~np.isnan(unet_series_1971) & ~np.isnan(unet_series_1771) & ~np.isnan(bicubic_series) & ~np.isnan(unet_series_combined)
    obs_series = obs_series[mask]
    unet_series_1971 = unet_series_1971[mask]
    unet_series_1771 = unet_series_1771[mask]
    bicubic_series = bicubic_series[mask]
    unet_series_combined = unet_series_combined[mask]

    # Common x grid for all CDFs
    all_series = [obs_series, unet_series_1971, unet_series_1771, bicubic_series, unet_series_combined]
    x_grid = np.linspace(np.nanmin(np.concatenate(all_series)), np.nanmax(np.concatenate(all_series)), 300)

    cdf_obs = empirical_cdf(obs_series, x_grid)
    cdf_unet_1971 = empirical_cdf(unet_series_1971, x_grid)
    cdf_unet_1771 = empirical_cdf(unet_series_1771, x_grid)
    cdf_bicubic = empirical_cdf(bicubic_series, x_grid)

    # Cramer-von Mises
    cvm_unet_1971 = cramervonmises_2samp(unet_series_1971, obs_series).statistic
    cvm_unet_1771 = cramervonmises_2samp(unet_series_1771, obs_series).statistic
    cvm_bicubic = cramervonmises_2samp(bicubic_series, obs_series).statistic

    # PSS
    pss_unet_1971 = perkins(cdf_unet_1971, cdf_obs)
    pss_unet_1771 = perkins(cdf_unet_1771, cdf_obs)
    pss_bicubic = perkins(cdf_bicubic, cdf_obs)


    #For combined
    cdf_combined = empirical_cdf(unet_series_combined, x_grid)
    cvm_combined = cramervonmises_2samp(unet_series_combined, obs_series).statistic
    pss_combined = perkins(cdf_combined, cdf_obs)

    plt.figure(figsize=(8,6))
    plt.plot(x_grid, cdf_obs, color="black", linewidth=2, label="Obs")
    plt.plot(x_grid, cdf_unet_1971, color="blue", linewidth=2, label=f"UNet 1971 (CvM={cvm_unet_1971:.3g}, PSS={pss_unet_1971:.3g})")
    plt.plot(x_grid, cdf_unet_1771, color="red", linewidth=2, label=f"UNet 1771 (CvM={cvm_unet_1771:.3g}, PSS={pss_unet_1771:.3g})")
    plt.plot(x_grid, cdf_bicubic, color="orange", linewidth=2, label=f"Bicubic (CvM={cvm_bicubic:.3g}, PSS={pss_bicubic:.3g})")
    plt.plot(x_grid, cdf_combined, color="green", linewidth=2, label=f"UNet Combined (CvM={cvm_combined:.3g}, PSS={pss_combined:.3g})")
    plt.axhline(1, color='gray', linestyle='--', linewidth=1)
    plt.title(f"{varname} CDF at {city_name} (lat={city_lat:.3f}, lon={city_lon:.3f})")
    plt.xlabel(varname)
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.tight_layout()
    output_path = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "Outputs" / f"CDF_{varname}_{city_name}_latlon_distance_UNet_pred.png"
    plt.savefig(str(output_path), dpi=500)
    plt.close()

if __name__ == "__main__":
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    city_coords = (47.3769, 8.5417) # Zürich
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
    unet_combined = xr.open_dataset(
    str(BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "models_UNet" / "UNet_Deterministic_Combined_Dataset" / "Combined_Dataset_Downscaled_Predictions_2011_2020.nc"),
    chunks={"time": 100}
)

    bicubic_paths = {
        "RhiresD": BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "RhiresD_step3_interp.nc",
        "TabsD":   BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "TabsD_step3_interp.nc",
        "TminD":   BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "TminD_step3_interp.nc",
        "TmaxD":   BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset" / "TmaxD_step3_interp.nc",
    }
    bicubic_path = bicubic_paths[hr_var]
    bicubic_ds = xr.open_dataset(str(bicubic_path), chunks={"time": 100}).sel(time=slice("2011-01-01", "2020-12-31"))

    plot_city_cdf_and_scores(
        city_coords,
        obs_ds[hr_var],
        unet_ds_1971[hr_var],
        unet_ds_1771[model_var],
        bicubic_ds[hr_var],
        unet_combined,
        varname=hr_var,
        city_name=city_name
    )