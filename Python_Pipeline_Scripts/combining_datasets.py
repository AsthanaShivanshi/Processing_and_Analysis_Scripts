import xarray as xr
from pathlib import Path
import os

BASE_DIR = Path(os.environ["BASE_DIR"])

var_map_1763 = {"precip": "pr", "temp": "tas", "tmin": "tasmin", "tmax": "tasmax"}
var_map_1971 = {"RhiresD": "pr", "TabsD": "tas", "TminD": "tasmin", "TmaxD": "tasmax"}

out_1971 = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "data_1971_2023" / "HR_files_full"
out_1763 = BASE_DIR / "raw_data" / "Reconstruction_UniBern_1763_2020"
combined_out = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "Combined_Dataset"
combined_out.mkdir(parents=True, exist_ok=True)


for std_var in ["pr", "tas", "tasmin", "tasmax"]:
    orig_1763 = [k for k, v in var_map_1763.items() if v == std_var][0]
    orig_1971 = [k for k, v in var_map_1971.items() if v == std_var][0]

    file_1763 = out_1763 / f"{orig_1763}_full.nc"
    file_1971 = out_1971 / f"{orig_1971}_full.nc"

    # Renaming
    ds_1763 = xr.open_dataset(file_1763).rename({orig_1763: std_var})
    ds_1971 = xr.open_dataset(file_1971).rename({orig_1971: std_var})

    # Adding 'source' dimension
    ds_1763 = ds_1763.expand_dims(source=["pretrain"])
    ds_1971 = ds_1971.expand_dims(source=["train"])

    # Concatenating along 'source'
    ds_combined = xr.concat([ds_1763, ds_1971], dim="source")
    print(f"Combined ds created for {std_var}")

    ds_combined.to_netcdf(combined_out / f"{std_var}_combined.nc")

print("Combined files with source dimension saved.")