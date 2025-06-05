import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def warm_wet_gridded(tmax, precip, precip_threshold=10.0,save=False, save_path=None):
    if not isinstance(tmax, xr.DataArray) or not isinstance(precip, xr.DataArray):
        raise TypeError("Both tmax and precip must be xarray.DataArray objects.")

    tmax, precip = xr.align(tmax, precip)

    time_dim = [dim for dim in tmax.dims if dim not in ['lat', 'lon']][0]

#Warm and wet conditions : 90th percentile TmaxD and precipitation thresholded by precip_threshold
    tmax_90 = tmax.quantile(0.9, dim=time_dim, skipna=True)
    warm = tmax > tmax_90
    wet = precip > precip_threshold
    warm_wet = warm & wet
    warm_wet_count = warm_wet.sum(dim=time_dim, skipna=True)

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([5.8, 10.6, 45.8, 47.9], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(tmax['lon'], tmax['lat'], warm_wet_count,
                       transform=ccrs.PlateCarree(), shading='auto')
    plt.colorbar(im, ax=ax, label='Number of Warm & Wet Days')
    plt.title(f'Warm and Wet Days (Tmax > 90th pct & Precip > {precip_threshold} mm)')
    plt.tight_layout()
    if save:
        if save_path is None:
            raise ValueError("save_path must be provided if save is True.")
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
