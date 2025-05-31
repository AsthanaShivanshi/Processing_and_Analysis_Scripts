import xarray as xr
import numpy as np
from pyproj import Transformer

def mu_sigma_pixelwise(city_coords, data_path, var_name):
    """
    Calculates the mean and standard deviation of a given variable at the grid cell
    closest to the specified city coordinates, assuming the data is pre-filtered.

    Parameters:
        city_coords (tuple): Latitude and longitude of the city (lat, lon).
        data_path (str): Path to the NetCDF file containing the variable.
        var_name (str): Name of the variable to analyze (e.g., 'TmaxD', 'TminD').

    Returns:
        tuple: Time series of the variable at the closest grid cell,
               mean and standard deviation of that series.
    """

    ds = xr.open_dataset(data_path)
    var = ds[var_name]

    E = ds["E"].values
    N = ds["N"].values
    E_grid, N_grid = np.meshgrid(E, N)

    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(E_grid, N_grid)

    # Compute distance to city
    city_lat, city_lon = city_coords
    dist = np.sqrt((lat - city_lat)**2 + (lon - city_lon)**2)
    lat_idx, lon_idx = np.unravel_index(np.argmin(dist), dist.shape)

    var_series = var[:, lat_idx, lon_idx]
    var_clean = var_series.dropna(dim="time")

    mu = np.mean(var_clean)
    sigma = np.std(var_clean, ddof=0)

    return var_clean, mu, sigma
