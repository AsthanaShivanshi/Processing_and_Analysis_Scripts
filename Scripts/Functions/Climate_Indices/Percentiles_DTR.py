import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm

def dtr_percentiles_gridded(
    tmax,
    tmin, 
    output_file=None,
    save=False,
    title_suffix=""
):
    if isinstance(tmax, str):
        ds_tmax = xr.open_dataset(tmax)
        tmax = ds_tmax['TmaxD']
    elif isinstance(tmax, xr.Dataset):
        tmax = tmax['TmaxD']
    elif isinstance(tmax, xr.DataArray):
        pass
    else:
        raise TypeError("tmax must be a path, Dataset, or DataArray")

    if isinstance(tmin, str):
        ds_tmin = xr.open_dataset(tmin)
        tmin = ds_tmin['TminD']
    elif isinstance(tmin, xr.Dataset):
        tmin = tmin['TminD']
    elif isinstance(tmin, xr.DataArray):
        pass
    else:
        raise TypeError("tmin must be a path, Dataset, or DataArray")
#DTR percentiles
    dtr = tmax - tmin
    dtr_5th = dtr.quantile(0.05, dim="time", skipna=True)
    dtr_95th = dtr.quantile(0.95, dim="time", skipna=True)

    lon = tmax['lon']
    lat = tmax['lat']

    fig, axs = plt.subplots(1, 2, figsize=(16, 6),
                            subplot_kw={'projection': ccrs.PlateCarree()})

    for ax, data, title in zip(
        axs,
        [dtr_5th, dtr_95th],
        [f'5th Percentile DTR {title_suffix}', f'95th Percentile DTR {title_suffix}']
    ):
        ax.set_extent([5.9, 10.5, 45.8, 47.8], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        norm = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)
        im = ax.pcolormesh(lon, lat, data,
                           transform=ccrs.PlateCarree(), shading='auto',
                           norm=norm, cmap='coolwarm')
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=axs, orientation='vertical', shrink=0.7, label='DTR (°C)')

    if save and output_file:
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Figure saved to {output_file}")

    plt.show()
