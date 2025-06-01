import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def warm_wet_gridded(file_path, precip_threshold=10.0):
    ds = xr.open_dataset(file_path)

    if 'latitude' in ds:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds:
        ds = ds.rename({'longitude': 'lon'})

    if not all(var in ds for var in ['TmaxD', 'RhiresD']):
        raise ValueError("Dataset must contain 'TmaxD' and 'RhiresD'.")

    time_dim = [dim for dim in ds['TmaxD'].dims if dim not in ['lat', 'lon']][0]

    tmax_90 = ds['TmaxD'].quantile(0.9, dim=time_dim, skipna=True)

    warm = ds['TmaxD'] > tmax_90

    wet = ds['RhiresD'] > precip_threshold

    warm_wet = warm & wet

    warm_wet_count = warm_wet.sum(dim=time_dim, skipna=True)

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(ds['lon'], ds['lat'], warm_wet_count,
                       transform=ccrs.PlateCarree(), shading='auto')
    plt.colorbar(im, ax=ax, label='Number of Warm & Wet Days')
    plt.title(f'Warm and Wet Days per Grid Cell (Tmax > 90th pct & Precip > {precip_threshold} mm)')
    plt.show()
