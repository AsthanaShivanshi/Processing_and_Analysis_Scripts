import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_frost_days(file_path, threshold=0.0):
    ds = xr.open_dataset(file_path)

    if 'latitude' in ds:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds:
        ds = ds.rename({'longitude': 'lon'})

    dims = list(ds['TminD'].dims)
    time_dim = [dim for dim in dims if dim not in ['lat', 'lon']]
    if len(time_dim) != 1:
        raise ValueError("Expected one time dimension other than lat/lon.")
    time_dim = time_dim[0]

    frost_days_mask = ds['TminD'] < threshold #Condition for frost days (Tmin < 0°C)

    frost_days_count = frost_days_mask.sum(dim=time_dim, skipna=True)

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(ds['lon'], ds['lat'], frost_days_count,
                       transform=ccrs.PlateCarree(), shading='auto')
    plt.colorbar(im, ax=ax, label='Number of Frost Days (Tmin < 0°C)')
    plt.title('Frost Days Count per Grid Cell')
    plt.show()
