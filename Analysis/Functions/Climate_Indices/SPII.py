import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def spii_gridded(precip, threshold=1.0, time_dim='time'):
    """ SPII for a gridded dataset.
    """
    # Masking wet days
    wet = precip.where(precip >= threshold)
    # Summing precip for each period
    wet_sum = wet.sum(dim=time_dim, skipna=True)
    wet_count = wet.count(dim=time_dim, skipna=True)
    # total precip on wet days / number of wet days
    spii = wet_sum / wet_count
    return spii

#Has to be seasonwise, #has to be calculated as Mean Annual


def plot_spii_seasons(precip, threshold=1.0, time_dim="time", extent=[5.8,10.6,45.8,47.9], save_path=None):

    seasons= {"MAM":[3,4,5],
              "JJA":[6,7,8],
              "SON":[9,10,11],
              "DJF":[12,1,2]}
    
    fig,axs=plt.subplots(2,2,figsize=(12,8), subplot_kw={'projection': None})
    axs= axs.flatten()

    for idx,(season, months) in enumerate(seasons.items()):
        if season=="DJF": #Have to handle the exception
            precip_season=precip.where((precip['time.month'] == 12) | (precip['time.month'] <= 2), drop=True)
        else:
            precip_season = precip.where(precip['time.month'].isin(months), drop=True)
        
        spii= spii_gridded(precip_season,threshold=threshold, time_dim=time_dim)
        im=axs[idx].pcolormesh(precip.lon, precip.lat, spii, cmap="viridis", shading="auto")
        axs[idx].set_title(f"Simple Precipitation Intensity Index (SPII) : {season}")
        axs[idx].set_xlabel("Longitude")
        axs[idx].set_ylabel("Latitude")
        plt.colorbar(im, ax=axs[idx], label="SPII (mm/day)")

    plt.suptitle("Seasonal Precipitation Intensity Index (SPII)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()