import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import scoreatpercentile

tabsd = xr.open_dataset('')['TabsD']

# rolling means: for calculating the 3 day means for excess heat index and acclimatization factor for heat stress index
tabsd_3day = tabsd.rolling(time=3, center=True).mean() #For EHI
tabsd_30day = tabsd.rolling(time=30, center=False).mean().shift(time=-3) #For acclimatization factor

#Climatological 95th pctl (1981–2010)
tabsd_clim = tabsd.sel(time=slice('1981-01-01', '2010-12-31'))
tabsd_clim_3day = tabsd_clim.rolling(time=3, center=True).mean()
doy_clim = tabsd_clim_3day['time'].dt.dayofyear

t95p_list = []
for day in range(1, 366):
    mask = np.isin(doy_clim, np.arange(day - 15, day + 16))
    vals = tabsd_clim_3day.sel(time=mask)
    p95 = scoreatpercentile(vals, 95, axis=0)
    t95p_list.append(p95)

t95p_array = np.stack(t95p_list, axis=0)
t95p = xr.DataArray(
    t95p_array,
    dims=["dayofyear", "lat", "lon"],
    coords={"dayofyear": np.arange(1, 366), "lat": tabsd.lat, "lon": tabsd.lon}
)

tabsd_3day = tabsd_3day.sel(time=tabsd_30day.time)
doy = tabsd_3day['time'].dt.dayofyear

t95p_matched = t95p.sel(dayofyear=doy)
t95p_matched['time'] = tabsd_3day['time']

# EHF : product of the two indices
EHI_sig = tabsd_3day - t95p_matched
EHI_accl = tabsd_3day - tabsd_30day
EHF = EHI_sig * xr.where(EHI_accl > 1, EHI_accl, 1)

# counting only heatwave days 
EHF = EHF.where(EHF['time.season'].isin(['JJA'])) 
heatwave_mask = EHF > 0

# Grouping and summing annually
heatwave_days_per_year = heatwave_mask.groupby('time.year').sum(dim='time')

heatwave_days_per_year.name = 'heatwave_day_count'

heatwave_days_per_year.to_netcdf('heatwave_days_per_year.nc')
