import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm

def dtr_percentiles_gridded(
    tmax_file,
    tmin_file,
    output_file=None,
    save=False,
    title_suffix=""
):
    ds_tmax = xr.open_dataset(tmax_file)
    ds_tmin = xr.open_dataset(tmin_file)

    if 'TmaxD' not in ds_tmax or 'TminD' not in ds_tmin:
        raise ValueError("No 'TmaxD' and/or 'TminD'.")

    dtr = ds_tmax['TmaxD'] - ds_tmin['TminD']
    dtr_5th = dtr.quantile(0.05, dim="time", skipna=True)
    dtr_95th = dtr.quantile(0.95, dim="time", skipna=True)

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
        norm= TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)
        im = ax.pcolormesh(ds_tmax['lon'], ds_tmax['lat'], data,
                           transform=ccrs.PlateCarree(), shading='auto', norm=norm, cmap='coolwarm')
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=axs, orientation='vertical', shrink=0.7, label='DTR (°C)')

    if save and output_file:
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Figure saved to {output_file}")

    plt.show()
