import sys
import os
from pathlib import Path

import xarray as xr
import numpy as np

try:
    BASE_DIR = Path(os.environ["BASE_DIR"])
except KeyError:
    raise EnvironmentError("BASE_DIR environment variable is not set. Did you source environment.sh?")

sys.path.append(str(BASE_DIR / "Scripts/Functions"))
from Gamma_KS_Test import Gamma_KS_gridded

season_name = sys.argv[1]
season_months_map = {
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5]
}

months = season_months_map.get(season_name)
if not months:
    raise ValueError(f"Invalid season name: {season_name}")

data_path = BASE_DIR / "Split_Data/Targets/train/rhiresd_targets_train.nc"
if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found at path: {data_path}")

ds = xr.open_dataset(data_path, chunks={"time": 100})

RhiresD = ds['RhiresD']
RhiresD = RhiresD.where(~np.isnan(RhiresD.lon) & ~np.isnan(RhiresD.lat))
season_mask = RhiresD["time"].dt.month.isin(months)
seasonal_data = RhiresD.sel(time=season_mask)

KS_Stat, p_val_ks_stat = Gamma_KS_gridded(
    seasonal_data,
    data_path=ds,
    block_size=10,
    season=season_name
)
