import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def txx_pmax_ratio(file_path):
    """
    Computing and plot TXx / Pmax index:
    - TXx =  Tmax max
    - Pmax = max daily precipitation
    - Ratio = TXx / Pmax (°C / mm)
    """
    ds = xr.open_dataset(file_path)

    txx = ds['TmaxD'].max(dim='time', skipna=True)
    pmax = ds['RhiresD'].max(dim='time', skipna=True)

    ratio = txx / pmax.where(pmax != 0) #to avoid division by zero error

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(ds['lon'], ds['lat'], ratio,
                       transform=ccrs.PlateCarree(), shading='auto')
    plt.colorbar(im, ax=ax, label='TXx / Pmax (°C/mm)')
    plt.title('TXx / Pmax Index per Grid Cell')
    plt.show()
