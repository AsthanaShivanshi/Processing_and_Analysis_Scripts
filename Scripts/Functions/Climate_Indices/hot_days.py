import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_hot_days(file_path, threshold=30.0):
    ds = xr.open_dataset(file_path)

    if 'latitude' in ds:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds:
        ds = ds.rename({'longitude': 'lon'})

    dims = list(ds['TmaxD'].dims)
    time_dim = [dim for dim in dims if dim not in ['lat', 'lon']]
    if len(time_dim) != 1:
        raise ValueError("Expected one time dimension other than lat/lon.")
    time_dim = time_dim[0]

    hot_days_mask = ds['TmaxD'] > threshold

    hot_days_count = hot_days_mask.sum(dim=time_dim, skipna=True)

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(ds['lon'], ds['lat'], hot_days_count,
                       transform=ccrs.PlateCarree(), shading='auto')
    plt.colorbar(im, ax=ax, label=f'Number of Hot Days (Tmax > {threshold}°C)')
    plt.title(f'Hot Days Count per Grid Cell (Tmax > {threshold}°C)')
    plt.show()
