import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def growing_degree_days_gridded(file_path, base_temp=5.0, season=None,save=False,save_path=None):
    """
    Computing and plotting Growing Degree Days.
    """
    ds = xr.open_dataset(file_path)
    time_dim = [dim for dim in ds['TabsD'].dims if dim not in ['lat', 'lon']][0]

    tabs = ds['TabsD']

    if season:
        tabs = tabs.sel(time=tabs['time.season'] == season)

    # Caluclating GDD: only days where TabsD > base_temp
    gdd_daily = (tabs - base_temp).where(tabs > base_temp, 0)

    # Summing GDD over time
    gdd_sum = gdd_daily.sum(dim=time_dim, skipna=True)

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(ds['lon'], ds['lat'], gdd_sum,
                       transform=ccrs.PlateCarree(), shading='auto')
    plt.colorbar(im, ax=ax, label=f'GDD (Base {base_temp}°C)')
    title = f'Growing Degree Days(based on threshold of 5 degrees C) per Grid Cell'
    if season:
        title += f' ({season})'
    plt.title(title)
    plt.savefig(save_path) if save and save_path else None
    plt.tight_layout()
    plt.show()
