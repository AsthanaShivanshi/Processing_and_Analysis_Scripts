import xarray as xr
from pathlib import Path
import os
import sys

BASE_DIR = Path(os.environ["BASE_DIR"])

# Only process tmin and tmax
var_list = ["tmin", "tmax"]
if len(sys.argv) > 1:
    idx = int(sys.argv[1])
    var_to_process = var_list[idx]
else:
    raise ValueError("Index for variable to process: missing (0 for tmin, 1 for tmax)")

var_map_1763 = {"tmin": "tmin", "tmax": "tmax"}
var_map_1971 = {"TminD": "tmin", "TmaxD": "tmax"}

out_1971 = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "data_1971_2023" / "HR_files_full"
out_1763 = BASE_DIR / "raw_data" / "Reconstruction_UniBern_1763_2020"
combined_out = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "Combined_Dataset"
combined_out.mkdir(parents=True, exist_ok=True)

std_var = var_to_process
orig_1763 = [k for k, v in var_map_1763.items() if v == std_var][0]
orig_1971 = [k for k, v in var_map_1971.items() if v == std_var][0]

file_1763 = out_1763 / f"{orig_1763}_1763_2020.nc"
file_1971 = out_1971 / f"{orig_1971}_1971_2023.nc"

chunk_size = 1000

ds_1763 = xr.open_dataset(file_1763, chunks={"time": chunk_size})
ds_1763 = ds_1763.rename({orig_1763: std_var})

ds_1971 = xr.open_dataset(file_1971, chunks={"time": chunk_size})
ds_1971 = ds_1971.rename({orig_1971: std_var})

vars_to_keep = [std_var, "lat", "lon", "time", "E", "N"]
ds_1763 = ds_1763[[v for v in vars_to_keep if v in ds_1763]]
ds_1971 = ds_1971[[v for v in vars_to_keep if v in ds_1971]]

for coord in ["lat", "lon"]:
    if coord in ds_1763 and coord not in ds_1763.coords:
        ds_1763 = ds_1763.set_coords(coord)
    if coord in ds_1971 and coord not in ds_1971.coords:
        ds_1971 = ds_1971.set_coords(coord)

# Concatenating and sorting by time to avoid slicing erors in the prepoorcessing pipeline
ds_merged = xr.concat([ds_1763, ds_1971], dim="time")
ds_merged = ds_merged.sortby("time")

for coord in ["lat", "lon"]:
    if coord in ds_merged and "time" in ds_merged[coord].dims:
        ds_merged[coord] = ds_merged[coord].isel(time=0)

ds_merged[std_var].attrs["coordinates"] = "lat lon"
if "coordinates" in ds_merged[std_var].encoding:
    del ds_merged[std_var].encoding["coordinates"]

if "N" in ds_merged:
    ds_merged["N"].attrs["units"] = "meters"
if "E" in ds_merged:
    ds_merged["E"].attrs["units"] = "meters"

print(f"{std_var} merged dimensions: {ds_merged.dims}")

ds_merged.to_netcdf(
    combined_out / f"{std_var}_merged.nc",
    engine="netcdf4"
)
print("Merged file saved.")

ds_1763.close()
ds_1971.close()