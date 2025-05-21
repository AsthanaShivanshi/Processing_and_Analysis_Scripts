import xarray as xr
import numpy as np
from pyproj import Transformer

def mu_sigma_pixelwise(city_coords, tabsd_path, rhiresd_path, threshold):
    """This function calculates the MLE-based mean and standard deviation at the pixel closest to the coordinates provided, assuming a Gaussian distribution of temperature on wet days."""

    # Load datasets
    ds1 = xr.open_dataset(tabsd_path)
    ds2 = xr.open_dataset(rhiresd_path)

    TabsD = ds1['TabsD']
    RhiresD = ds2['RhiresD']

    # Get grid coordinates (E/N)
    E = ds1["E"].values
    N = ds1["N"].values
    E_grid, N_grid = np.meshgrid(E, N)

    # Transform to lat/lon
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(E_grid, N_grid)

    # Get target city coordinates
    city_lat, city_lon = city_coords

    # Compute distance
    dist = np.sqrt((lat - city_lat)**2 + (lon - city_lon)**2)
    min_dist_index = np.unravel_index(np.argmin(dist), dist.shape)
    closest_lat_idx, closest_lon_idx = min_dist_index

    # Mask NaNs from original dataset
    mask = np.isnan(TabsD) | np.isnan(RhiresD)
    TabsD_gridded = TabsD.where(~mask)
    RhiresD_gridded = RhiresD.where(~mask)

    # Extract time series at closest grid cell
    tabsd = TabsD_gridded[:, closest_lat_idx, closest_lon_idx]
    rhiresd = RhiresD_gridded[:, closest_lat_idx, closest_lon_idx]

    # Wet day mask
    wet_mask = rhiresd >= threshold
    tabsd_wet = tabsd[wet_mask]
    rhiresd_wet = rhiresd[wet_mask]

    # Compute stats
    mu_tabs = np.mean(tabsd_wet)
    sigma_tabs = np.std(tabsd, ddof=0)

    return tabsd_wet, rhiresd_wet, mu_tabs, sigma_tabs
