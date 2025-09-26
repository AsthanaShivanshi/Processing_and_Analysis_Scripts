import os
import sys
sys.path.append('../Prelim_Stats')
import config
import numpy as np
import xarray as xr
import pyproj
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from scipy.stats import probplot
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import multivariate_normal
from pyproj import Transformer
from statsmodels.graphics.gofplots import qqplot


def select_nearest_grid_cell(dataset, target_lat, target_lon, var_name=None):
    
    if 'lat' in dataset.coords and 'lon' in dataset.coords:
        lat_vals = dataset['lat'].values
        lon_vals = dataset['lon'].values
        
        if lat_vals.ndim == 1 and lon_vals.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
        else:
            lat_grid = lat_vals
            lon_grid = lon_vals
            
    elif 'N' in dataset.coords and 'E' in dataset.coords:
        E = dataset["E"].values  
        N = dataset["N"].values  
        E_grid, N_grid = np.meshgrid(E, N)
        
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
        lon_grid, lat_grid = transformer.transform(E_grid, N_grid)
        
    elif 'x' in dataset.coords and 'y' in dataset.coords:
        x = dataset["x"].values
        y = dataset["y"].values
        x_grid, y_grid = np.meshgrid(x, y)
        
        lon_grid = x_grid
        lat_grid = y_grid
        
    else:
        raise ValueError("Dataset must have either 'lat'/'lon', 'N'/'E', or 'x'/'y' coordinates")
    
    dist = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
    min_dist_index = np.unravel_index(np.argmin(dist), dist.shape)
    
    closest_lat = lat_grid[min_dist_index]
    closest_lon = lon_grid[min_dist_index]
    min_distance = dist[min_dist_index]
    
    lat_idx, lon_idx = min_dist_index
    
    # Prepare result dictionary
    result = {
        'lat_idx': lat_idx,
        'lon_idx': lon_idx,
        'closest_lat': closest_lat,
        'closest_lon': closest_lon,
        'distance': min_distance
    }
    
    if var_name is not None:
        if var_name not in dataset:
            available_vars = list(dataset.data_vars.keys())
            raise ValueError(f"Variable '{var_name}' not found. Available variables: {available_vars}")
        
        if 'lat' in dataset.coords and 'lon' in dataset.coords:
            if dataset[var_name].dims[-2:] == ('lat', 'lon'):
                data_series = dataset[var_name].isel(lat=lat_idx, lon=lon_idx)
            else:
                dims = dataset[var_name].dims
                if 'y' in dims and 'x' in dims:
                    data_series = dataset[var_name].isel(y=lat_idx, x=lon_idx)
                else:
                    data_series = dataset[var_name].isel({dims[-2]: lat_idx, dims[-1]: lon_idx})
        else:
            # Use N/E or x/y dimensions
            dims = dataset[var_name].dims
            if 'N' in dims and 'E' in dims:
                data_series = dataset[var_name].isel(N=lat_idx, E=lon_idx)
            elif 'y' in dims and 'x' in dims:
                data_series = dataset[var_name].isel(y=lat_idx, x=lon_idx)
            else:
                data_series = dataset[var_name].isel({dims[-2]: lat_idx, dims[-1]: lon_idx})
        
        result['data'] = data_series
    
    print(f"Target: ({target_lat:.4f}, {target_lon:.4f})")
    print(f"Closest grid cell: ({closest_lat:.4f}, {closest_lon:.4f})")
    print(f"Distance: {min_distance:.4f} degrees")
    print(f"Grid indices: lat_idx={lat_idx}, lon_idx={lon_idx}")
    
    if var_name:
        print(f"Variable '{var_name}' extracted")
        print(f"Time series shape: {result['data'].shape}")
        print(f"Data range: {result['data'].min().values:.2f} to {result['data'].max().values:.2f}")
    
    return result

