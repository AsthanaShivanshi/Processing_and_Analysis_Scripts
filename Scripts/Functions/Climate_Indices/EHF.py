import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import scoreatpercentile

def ehf_days_gridded(tabsd, analysis_period=('2019-01-01', '2023-12-31'),
                     clim_period=('1981-01-01', '2010-12-31'), save_path=None):

    tabsd_analysis = tabsd.sel(time=slice(*analysis_period))
    tabsd_clim = tabsd.sel(time=slice(*clim_period))

    # Rolling avgs
    tabsd_3day = tabsd_analysis.rolling(time=3, center=True).mean()
    tabsd_30day = tabsd_analysis.rolling(time=30, center=False).mean().shift(time=-3)

    # Climatology 3-day rolling 
    tabsd_clim_3day = tabsd_clim.rolling(time=3, center=True).mean()
    doy_clim = tabsd_clim_3day['time'].dt.dayofyear

    #  95th percentile for each doy ±15-day window
    t95p_list = []
    for day in range(1, 366):
        window_days = np.arange(day - 15, day + 16)
        window_days = np.where(window_days < 1, window_days + 365, window_days)
        window_days = np.where(window_days > 365, window_days - 365, window_days)
        mask = np.isin(doy_clim, window_days)
        vals = tabsd_clim_3day.sel(time=mask)
        p95 = scoreatpercentile(vals, 95, axis=0)
        t95p_list.append(p95)

    t95p_array = np.stack(t95p_list, axis=0)
    t95p = xr.DataArray(
        t95p_array,
        dims=["dayofyear", "lat", "lon"],
        coords={"dayofyear": np.arange(1, 366), "lat": tabsd.lat, "lon": tabsd.lon}
    )

    # Matchingh doy from analysis period
    doy = tabsd_3day['time'].dt.dayofyear
    t95p_matched = t95p.sel(dayofyear=doy)
    t95p_matched['time'] = tabsd_3day['time']

    # EHF 
    EHI_sig = tabsd_3day - t95p_matched
    EHI_accl = tabsd_3day - tabsd_30day
    EHF = EHI_sig * xr.where(EHI_accl > 1, EHI_accl, 1)

    # Only JJA
    EHF = EHF.where(EHF['time.season'].isin(['JJA']))
    heatwave_mask = EHF > 0

    # Summing heatwave days per grid cell
    heatwave_days_total = heatwave_mask.sum(dim='time', skipna=True)

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([5.8, 10.6, 45.8, 47.9], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    im = ax.pcolormesh(tabsd['lon'], tabsd['lat'], heatwave_days_total,
                       transform=ccrs.PlateCarree(), shading='auto', cmap='Reds')
    plt.colorbar(im, ax=ax, label='Total Heatwave Days (JJA)')
    plt.title(f'Total Heatwave Days per Grid Cell ({analysis_period[0]} to {analysis_period[1]}, EHF > 0, JJA)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()

    return heatwave_days_total
