from KS_gridded import Kalmogorov_Smirnov_gridded
import xarray as xr
import numpy as np

def run_ks_tests():
    # Load datasets
    ds1 = xr.open_dataset("../../data/processed/Bicubic/Train/targets_tas_masked_train.nc", chunks={"time": 100})
    ds2 = xr.open_dataset("../../data/processed/Bicubic/Train/targets_precip_masked_train.nc", chunks={"time": 100})

    TabsD = ds1['TabsD']
    RhiresD = ds2['RhiresD']

    # Apply mask for NaN in lat/lon
    lon = TabsD.lon
    lat = TabsD.lat
    mask = np.isnan(lon) | np.isnan(lat)
    TabsD_gridded = TabsD.where(~mask)
    RhiresD_gridded = RhiresD.where(~mask)

    # Only wet days
    TabsD_wet = TabsD_gridded.where(RhiresD_gridded >= 0.1)
    RhiresD_wet = RhiresD_gridded.where(RhiresD_gridded >= 0.1)

    # Define seasons
    seasons = {
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5]
    }

    # Loop over seasons
    for season_name, months in seasons.items():
        mask_months = TabsD_wet['time'].dt.month.isin(months)
        TabsD_wet_season = TabsD_wet.sel(time=mask_months)
        RhiresD_wet_season = RhiresD_wet.sel(time=mask_months)

        # Compute seasonal mean and std
        Mu_TabsD_season = TabsD_wet_season.mean(dim="time", skipna=True)
        Sigma_TabsD_season = TabsD_wet_season.std(dim="time", ddof=0, skipna=True)

        # Run KS test
        KS_Stat, p_val_ks_stat = Kalmogorov_Smirnov_gridded(
            TabsD_wet_season, 
            Mu_TabsD_season, 
            Sigma_TabsD_season, 
            data_path=ds1,
            block_size=20,
            season=season_name
        )

        print(f"Finished KS Test and plotting for {season_name}")

if __name__ == "__main__":
    run_ks_tests()

