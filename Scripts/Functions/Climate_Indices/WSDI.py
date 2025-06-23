import xarray as xr
import pandas as pd
import numpy as np

def TXin90(tmax, base_period=("1981-01-01", "2010-12-31"), window=5):

    "Calculating 90th pctl of tmax centered on a 5 day window for each calendar day, over the climatological period"
    #Selecting the base period and calculating the 90th pctl, with padding for leap years
    tmax_base= tmax.sel(time=slice(*base_period)).groupby("time.dayofyear")

    TXin90= []
    for doy in range(1,367):
        window_days= [(doy + i -window/2 -1) %366+1 for i in range(window)]
        vals= xr.concat([tmax_base.get_group(day) for day in window_days if day in tmax_base.groups], dim="time")
        TXin90.append(vals.quantile(0.9, dim="time"))
    TXin90= xr.concat(TXin90, dim="dayofyear")
    TXin90["dayofyear"]= np.arange(1, 367)
    return TXin90



def wsdi_gridded(tmax,TXin90, min_spell=6):
    """Gridded Warm Spell Duration Index (WSDI) calc"""

    years= np.unique(tmax["time.year"].values)
    wsdi=[]
    for year in years:
        tmax_year= tmax.sel(time=str(year))
        doy= tmax_year["time.dt.dayofyear"]
        thresh= TXin90.sel(dayofyear=doy)
    #Boolean mask for warm spell days for counting gridwise later
        warm=(tmax_year>thresh)
        arr= warm.values
        out= np.zeros(arr.shape[1:], dtype= int)
        for i in range (arr.shape[1]):
            for j in range(arr.shape[2]):
                mask= arr [:,i,j]
                #Running of consecutive true boolean mask values 
                count=0
                total=0
                for val in mask:
                    if val:
                        count=count+1
                    else:
                        if count>=min_spell:
                            total=total+count
                        count=0
                if count>=min_spell:
                    total =total+count
                out[i,j]=total 
        wsdi.append(xr.DataArray(out,coords=[tmax.lat,tmax.lon],dims=["lat", "lon"]))
    wsdi=xr.concat(wsdi, dim="year")
    wsdi["year"]=years
    return wsdi

#Might be required : Mean  wsdi for each grid cell, or a 2D estimate of year to year variability. 