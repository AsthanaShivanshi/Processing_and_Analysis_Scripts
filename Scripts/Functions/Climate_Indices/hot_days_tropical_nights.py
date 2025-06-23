import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def hot_days_gridded(ds, threshold=30.0, title='Hot Days per Grid Cell',save=False,save_path='../../Outputs/hot_days_map.png'):
    if 'latitude' in ds:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds:
        ds = ds.rename({'longitude': 'lon'})

    if 'TmaxD' not in ds:
        raise ValueError("Dataset must contain 'TmaxD'.")

    time_dim = [dim for dim in ds['TmaxD'].dims if dim not in ['lat', 'lon']][0]
    tropical_nights = (ds['TmaxD'] > threshold).sum(dim=time_dim, skipna=True)

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([5.8, 10.6, 45.8, 47.9], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(ds['lon'], ds['lat'], tropical_nights,
                       transform=ccrs.PlateCarree(),
                       shading='auto', cmap='coolwarm', vmin=0, vmax=100)

    cbar = plt.colorbar(im, ax=ax, label=f'Number of Hot Days (Tmax > {threshold}°C)')
    cbar.set_ticks(np.arange(0, 101, 10))

    plt.title(title)
    plt.tight_layout()
    
    if save:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()



def tropical_nights_gridded(ds, threshold=20.0, title='Tropical Nights per Grid Cell',save=False,save_path='../../Outputs/tropical_nights_map.png'):
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
    ax.set_extent([5.8, 10.6, 45.8, 47.9], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(ds['lon'], ds['lat'], tropical_nights,
                       transform=ccrs.PlateCarree(),
                       shading='auto', cmap='coolwarm', vmin=0, vmax=100)

    cbar = plt.colorbar(im, ax=ax, label=f'Number of Tropical Nights (Tmin > {threshold}°C)')
    cbar.set_ticks(np.arange(0, 101, 10))
    plt.title(title)
    plt.tight_layout()

    if save:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

