import xarray as xr
from pathlib import Path
import os
import sys
import numpy as np

BASE_DIR = Path(os.environ["BASE_DIR"])

var_list = ["pr", "tas", "tasmin", "tasmax"]
if len(sys.argv) > 1:
    idx = int(sys.argv[1])
    var_to_process = var_list[idx]
else:
    raise ValueError("Index for variable to process: missing (between 0-3)")

var_map_1763 = {"precip": "pr", "temp": "tas", "tmin": "tasmin", "tmax": "tasmax"}
var_map_1971 = {"RhiresD": "pr", "TabsD": "tas", "TminD": "tasmin", "TmaxD": "tasmax"}

out_1971 = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "data_1971_2023" / "HR_files_full"
out_1763 = BASE_DIR / "raw_data" / "Reconstruction_UniBern_1763_2020"
combined_out = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "Combined_Dataset"
combined_out.mkdir(parents=True, exist_ok=True)

std_var=var_to_process

orig_1763 = [k for k, v in var_map_1763.items() if v == std_var][0]
orig_1971 = [k for k, v in var_map_1971.items() if v == std_var][0]

file_1763 = out_1763 / f"{orig_1763}_1763_2020.nc"
file_1971 = out_1971 / f"{orig_1971}_1971_2023.nc"

ds_1763 = xr.open_dataset(file_1763).rename({orig_1763: std_var})
ds_1971 = xr.open_dataset(file_1971).rename({orig_1971: std_var})

# Interpolating 1971 to the 1763 grid (E=265,N=370)
ds_1971_interp = ds_1971.interp(E=ds_1763['E'], N=ds_1763['N'], method="linear")

# Drop lat/lon ::: facing mismatch issues, depending on ds
for coord in ["lat", "lon"]:
    if coord in ds_1763.coords:
        ds_1763 = ds_1763.drop_vars(coord)
    if coord in ds_1971_interp.coords:
        ds_1971_interp = ds_1971_interp.drop_vars(coord)

print(f"{std_var} 1763 dims: {ds_1763.dims}")
print(f"{std_var} 1971 dims: {ds_1971_interp.dims}")

# Duplicate timesteps allowed
ds_merged = xr.concat([ds_1763, ds_1971_interp], dim="time")
print(f"Concatenated ds created for {std_var}")
print(f"{std_var} merged dimensions: {ds_merged.dims}")

ds_merged.to_netcdf(combined_out / f"{std_var}_merged.nc")

print("Merged file (with duplicate times in overlap) saved.")