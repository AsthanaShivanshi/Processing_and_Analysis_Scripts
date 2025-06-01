import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def hot_days_gridded(file_path, threshold=30.0):
    ds = xr.open_dataset(file_path)

    if 'latitude' in ds:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds:
        ds = ds.rename({'longitude': 'lon'})

    if 'TmaxD' not in ds:
        raise ValueError("Dataset must contain 'TmaxD'.")

    time_dim = [dim for dim in ds['TmaxD'].dims if dim not in ['lat', 'lon']][0]

    hot_days = (ds['TmaxD'] > threshold).sum(dim=time_dim, skipna=True)

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(ds['lon'], ds['lat'], hot_days,
                       transform=ccrs.PlateCarree(), shading='auto')
    plt.colorbar(im, ax=ax, label=f'Number of Hot Days (Tmax > {threshold}°C)')
    plt.title('Hot Days per Grid Cell')
    plt.show()

def tropical_nights_gridded(file_path, threshold=20.0):
    ds = xr.open_dataset(file_path)

    if 'latitude' in ds:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds:
        ds = ds.rename({'longitude': 'lon'})

    if 'TminD' not in ds:
        raise ValueError("Dataset must contain 'TminD'.")

    time_dim = [dim for dim in ds['TminD'].dims if dim not in ['lat', 'lon']][0]

    tropical_nights = (ds['TminD'] > threshold).sum(dim=time_dim, skipna=True)

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(ds['lon'], ds['lat'], tropical_nights,
                       transform=ccrs.PlateCarree(), shading='auto')
    plt.colorbar(im, ax=ax, label=f'Number of Tropical Nights (Tmin > {threshold}°C)')
    plt.title('Tropical Nights per Grid Cell')
    plt.show()
