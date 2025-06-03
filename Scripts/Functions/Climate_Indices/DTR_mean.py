import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def dtr_mean_gridded(
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
        raise TypeError("tmax must be a file path, Dataset, or DataArray")

    if isinstance(tmin, str):
        ds_tmin = xr.open_dataset(tmin)
        tmin = ds_tmin['TminD']
    elif isinstance(tmin, xr.Dataset):
        tmin = tmin['TminD']
    elif isinstance(tmin, xr.DataArray):
        pass
    else:
        raise TypeError("tmin must be a file path, Dataset, or DataArray")


    dtr = tmax - tmin
    mean_dtr = dtr.mean(dim="time", skipna=True)


    lon = tmax['lon']
    lat = tmax['lat']

    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([5.9, 10.5, 45.8, 47.8], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(lon, lat, mean_dtr,
                       transform=ccrs.PlateCarree(), shading='auto',
                       vmin=0, vmax=15, cmap='coolwarm')
    plt.colorbar(im, ax=ax, label='Mean Diurnal Temperature Range (°C)')
    plt.title(f'Mean DTR over Switzerland {title_suffix}')

    if save and output_file:
        plt.savefig(output_file)
        print(f"Mean DTR saved to {output_file}")

    plt.show()
