import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def dtr_mean_gridded(
    tmax_file,
    tmin_file,
    output_file=None,
    save=False,
    title_suffix=""
):
    ds_tmax = xr.open_dataset(tmax_file)
    ds_tmin = xr.open_dataset(tmin_file)

    if 'TmaxD' not in ds_tmax or 'TminD' not in ds_tmin:
        raise ValueError("Datasets must contain 'TmaxD' and 'TminD' respectively.")

    #  DTR and mean over time
    dtr = ds_tmax['TmaxD'] - ds_tmin['TminD']
    mean_dtr = dtr.mean(dim="time", skipna=True)

    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([5.9, 10.5, 45.8, 47.8], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(ds_tmax['lon'], ds_tmax['lat'], mean_dtr,
                       transform=ccrs.PlateCarree(), shading='auto',vmin=0, vmax=15, cmap='coolwarm')
    plt.colorbar(im, ax=ax, label='Mean Diurnal Temperature Range (°C)')
    plt.title(f'Mean DTR over Switzerland {title_suffix}')

    
    if save:
        plt.savefig(output_file)
        print(f"Mean DTR saved to {output_file}")
        
    plt.show()