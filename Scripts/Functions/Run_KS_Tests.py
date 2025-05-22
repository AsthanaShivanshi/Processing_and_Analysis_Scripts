import sys
import os
from pathlib import Path
import xarray as xr
import numpy as np

from KS_gridded import Kalmogorov_Smirnov_gridded

if __name__ == "__main__":
    BASE_DIR = Path(os.environ["BASE_DIR"])

    season_name = sys.argv[1]
    season_months_map = {
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5]
    }
    months = season_months_map.get(season_name)
    if not months:
        raise ValueError(f"Invalid season: {season_name}")

    data_path = BASE_DIR / "Split_Data/Targets/train/tabsd_targets_train.nc"
    if not data_path.exists():
        raise FileNotFoundError(f"TmaxD NetCDF file not found at: {data_path}")

    ds = xr.open_dataset(data_path, chunks={"time": 100})
    TabsD = ds["TabsD"]

    TabsD_season = TabsD.sel(time=TabsD["time"].dt.month.isin(months))

    Mu_TabsD = TabsD_season.mean(dim="time", skipna=True)
    Sigma_TabsD = TabsD_season.std(dim="time", ddof=0, skipna=True)

    KS_Stat, p_val_ks_stat = Kalmogorov_Smirnov_gridded(
        TabsD_season,
        Mu_TabsD,
        Sigma_TabsD,
        data_path=ds,
        block_size=20,
        season=season_name
    )
