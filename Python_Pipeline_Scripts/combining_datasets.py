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

chunk_size = 300

ds_1763 = xr.open_dataset(file_1763, chunks={"time": chunk_size})
if orig_1763 == "precip":
    ds_1763["precip"] = xr.where(ds_1763["precip"] < 0, 0, ds_1763["precip"])
ds_1763 = ds_1763.rename({orig_1763: std_var})

ds_1971 = xr.open_dataset(file_1971, chunks={"time": chunk_size})
if orig_1971 == "RhiresD":
    ds_1971["RhiresD"] = xr.where(ds_1971["RhiresD"] < 0, 0, ds_1971["RhiresD"])
ds_1971 = ds_1971.rename({orig_1971: std_var})

vars_to_keep = [std_var, "lat", "lon", "time", "E", "N"]
ds_1763 = ds_1763[[v for v in vars_to_keep if v in ds_1763]]
ds_1971 = ds_1971[[v for v in vars_to_keep if v in ds_1971]]

for coord in ["lat", "lon"]:
    if coord in ds_1763 and coord not in ds_1763.coords:
        ds_1763 = ds_1763.set_coords(coord)
    if coord in ds_1971 and coord not in ds_1971.coords:
        ds_1971 = ds_1971.set_coords(coord)


# 1971-2000 (train), 2001-2010 (val) from ds_1971
# 1771-1980 (train), 1981-2010 (val) from ds_1763

# Train split
ds_1971_train = ds_1971.sel(time=slice("1971-01-01", "2000-12-31"))
ds_1763_train = ds_1763.sel(time=slice("1771-01-01", "2000-12-31"))
ds_train = xr.concat([ds_1763_train, ds_1971_train], dim="time")

# Validation split
ds_1971_val = ds_1971.sel(time=slice("2001-01-01", "2010-12-31"))
ds_1763_val = ds_1763.sel(time=slice("2001-01-01", "2010-12-31"))
ds_val = xr.concat([ds_1763_val, ds_1971_val], dim="time")

for ds_merged in [ds_train, ds_val]:
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

print(f"{std_var} train split dimensions: {ds_train.dims}")
print(f"{std_var} validation split dimensions: {ds_val.dims}")

ds_train.to_netcdf(
    combined_out / f"{std_var}_train_merged.nc",
    engine="netcdf4"
)
ds_val.to_netcdf(
    combined_out / f"{std_var}_val_merged.nc",
    engine="netcdf4"
)
print("Train and validation merged files saved.")

ds_1763.close()
ds_1971.close()
ds_train.close()
ds_val.close()