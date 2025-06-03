import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def growing_degree_days_gridded(data, base_temp=5.0, season=None, save=False, save_path=None):

    if isinstance(data, str):
        ds = xr.open_dataset(data)
        tabs = ds['TabsD']
    elif isinstance(data, xr.Dataset):
        tabs = data['TabsD']
    elif isinstance(data, xr.DataArray):
        tabs = data
    else:
        raise TypeError("data must be a file path, xarray Dataset, or DataArray")

    if season:
        tabs = tabs.sel(time=tabs['time.season'] == season)

    time_dim = [dim for dim in tabs.dims if dim not in ['lat', 'lon']][0]

    gdd_daily = (tabs - base_temp).where(tabs > base_temp, 0)
    gdd_sum = gdd_daily.sum(dim=time_dim, skipna=True)

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([5.8, 10.6, 45.8, 47.9], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(tabs['lon'], tabs['lat'], gdd_sum,
                       transform=ccrs.PlateCarree(), shading='auto')
    plt.colorbar(im, ax=ax, label=f'GDD (Base {base_temp}°C)')

    title = 'Growing Degree Days (Base 5°C)'
    if season:
        title += f' — {season}'
    plt.title(title)
    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")

    plt.show()
