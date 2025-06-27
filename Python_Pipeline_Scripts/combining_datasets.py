import xarray as xr
from pathlib import Path
import os
import sys

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

    # Rename
ds_1763 = xr.open_dataset(file_1763).rename({orig_1763: std_var})
ds_1971 = xr.open_dataset(file_1971).rename({orig_1971: std_var})

print(f"{std_var} 1763 dims: {ds_1763.dims}")
print(f"{std_var} 1971 dims: {ds_1971.dims}")

# 'source' dim
ds_1763 = ds_1763.expand_dims(source=["pretrain"])
ds_1971 = ds_1971.expand_dims(source=["train"])


#For handling E/N and lat/lon problem
for coord in ["lat", "lon"]:
    if coord in ds_1763.coords:
        ds_1763 = ds_1763.drop_vars(coord)
    if coord in ds_1971.coords:
        ds_1971 = ds_1971.drop_vars(coord)

print(f"{std_var} 1763 dims: {ds_1763.dims}")
print(f"{std_var} 1971 dims: {ds_1971.dims}")

    # Concatenating along source due to overlapping time coords after 1971
ds_combined = xr.concat([ds_1763, ds_1971], dim="source")
print(f"Combined ds created for {std_var}")
print(f"{std_var} combined dimensions: {ds_combined.dims}")

ds_combined.to_netcdf(combined_out / f"{std_var}_combined.nc")

print("Combined files with source dimension saved.")