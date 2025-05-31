import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_percentile_range(file_path):
    ds = xr.open_dataset(file_path)
    
    if 'latitude' in ds:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds:
        ds = ds.rename({'longitude': 'lon'})
    dims = list(ds['RhiresD'].dims)
    other_dims = [dim for dim in dims if dim not in ['lat', 'lon']]
    
    #  5th and 95th percentiles per grid cell
    p5 = ds['RhiresD'].quantile(0.05, dim=other_dims, skipna=True)
    p95 = ds['RhiresD'].quantile(0.95, dim=other_dims, skipna=True)
    
    # Computing the range
    percentile_range = p95 - p5
   
    # Plotting
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(ds['lon'], ds['lat'], percentile_range,
                       transform=ccrs.PlateCarree(), shading='auto')
    plt.colorbar(im, ax=ax, label='95th - 5th Percentile Range of RhiresD')

    plt.title('Grid-wise 5th to 95th Percentile Range of RhiresD')
    plt.show()
