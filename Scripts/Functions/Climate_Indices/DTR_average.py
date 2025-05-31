import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def mean_dtr_gridded(file_path):
    ds = xr.open_dataset(file_path)

    if 'latitude' in ds:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds:
        ds = ds.rename({'longitude': 'lon'})

    if not all(var in ds for var in ['TmaxD', 'TminD']):
        raise ValueError("Dataset must contain 'TmaxD' and 'TminD'.")

    dtr = ds['TmaxD'] - ds['TminD']

    dims = list(dtr.dims)
    time_dim = [dim for dim in dims if dim not in ['lat', 'lon']][0]

    mean_dtr = dtr.mean(dim=time_dim, skipna=True)

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(ds['lon'], ds['lat'], mean_dtr,
                       transform=ccrs.PlateCarree(), shading='auto')
    plt.colorbar(im, ax=ax, label='Mean Diurnal Temperature Range (°C)')
    plt.title('Average Diurnal Temperature Range per Grid Cell')
    plt.show()
