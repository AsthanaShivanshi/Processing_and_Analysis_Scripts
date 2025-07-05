import xarray as xr
from pathlib import Path
import os
import sys

BASE_DIR = Path(os.environ["BASE_DIR"])

var_list = ["precip", "temp", "tmin", "tmax"]
if len(sys.argv) > 1:
    idx = int(sys.argv[1])
    var_to_process = var_list[idx]
else:
    raise ValueError("Index for variable to process: missing (between 0-3)")

var_map_1763 = {"precip": "precip", "temp": "temp", "tmin": "tmin", "tmax": "tmax"}
var_map_1971 = {"RhiresD": "precip", "TabsD": "temp", "TminD": "tmin", "TmaxD": "tmax"}

out_1971 = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "data_1971_2023" / "HR_files_full"
out_1763 = BASE_DIR / "raw_data" / "Reconstruction_UniBern_1763_2020"
combined_out = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "Combined_Dataset"
combined_out.mkdir(parents=True, exist_ok=True)

std_var = var_to_process
orig_1763 = [k for k, v in var_map_1763.items() if v == std_var][0]
orig_1971 = [k for k, v in var_map_1971.items() if v == std_var][0]

file_1763 = out_1763 / f"{orig_1763}_1763_2020.nc"
file_1971 = out_1971 / f"{orig_1971}_1971_2023.nc"

# Harmonisign var
ds_1763 = xr.open_dataset(file_1763)
if orig_1763 == "precip":
    ds_1763["precip"] = xr.where(ds_1763["precip"] < 0, 0, ds_1763["precip"])
ds_1763 = ds_1763.rename({orig_1763: std_var})

ds_1971 = xr.open_dataset(file_1971)
if orig_1971 == "RhiresD":
    ds_1971["RhiresD"] = xr.where(ds_1971["RhiresD"] < 0, 0, ds_1971["RhiresD"])
ds_1971 = ds_1971.rename({orig_1971: std_var})


for coord in ["lat", "lon"]:
    if coord in ds_1763 and coord not in ds_1763.coords:
        ds_1763 = ds_1763.set_coords(coord)
    if coord in ds_1971 and coord not in ds_1971.coords:
        ds_1971 = ds_1971.set_coords(coord)

ds_merged = xr.concat([ds_1763, ds_1971], dim="time")
ds_merged = ds_merged.sortby("time")

#Setting lat lon
for coord in ["lat", "lon"]:
    if coord in ds_merged and coord not in ds_merged.coords:
        ds_merged = ds_merged.set_coords(coord)

#Squeezing out time from lat lon
for coord in ["lat", "lon"]:
    if coord in ds_merged and "time" in ds_merged[coord].dims:
        ds_merged[coord] = ds_merged[coord].isel(time=0)

for v in list(ds_merged.data_vars) + list(ds_merged.coords):
    if hasattr(ds_merged[v], "attrs"):
        ds_merged[v].attrs.pop("grid_mapping", None)
        ds_merged[v].attrs.pop("coordinates", None)
if "grid_mapping" in ds_merged.attrs:
    del ds_merged.attrs["grid_mapping"]

# swiss_lv95_coordinates var dropped
if "swiss_lv95_coordinates" in ds_merged:
    ds_merged = ds_merged.drop_vars("swiss_lv95_coordinates")

ds_merged[std_var].attrs["coordinates"] = "lat lon"
print(f"{std_var} merged dimensions: {ds_merged.dims}")
ds_merged.to_netcdf(combined_out / f"{std_var}_merged.nc")
print("Merged file (with duplicate times in overlap) saved.")