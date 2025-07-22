import os
import json
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
from pyproj import Transformer, datadir
import time
import subprocess
import tempfile


BASE_DIR = Path(os.environ["BASE_DIR"])
INPUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "Combined_Dataset"
OUTPUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Combined_Chronological_Dataset_Revised"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
proj_path = os.environ.get("PROJ_LIB") or "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvNew/share/proj"
os.environ["PROJ_LIB"] = proj_path
datadir.set_data_dir(proj_path)

VAR_RENAME_MAP = {
    "RhiresD": "precip",
    "TabsD": "temp",
    "TminD": "tmin",
    "TmaxD": "tmax"
}

def save(ds, path):
    encoding = {v: {"_FillValue": np.nan} for v in ds.data_vars}
    ds.to_netcdf(str(path), encoding=encoding)
    ds.close()

def promote_latlon(ds, varname):
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    def transform_coords(e, n):
        lon, lat = transformer.transform(e, n)
        return np.stack([lon, lat], axis=0)
    E, N = ds["E"], ds["N"]
    EE, NN = xr.broadcast(E, N)
    transformed = xr.apply_ufunc(
        transform_coords, EE, NN,
        input_core_dims=[["N", "E"], ["N", "E"]],
        output_core_dims=[["coord_type", "N", "E"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    lon, lat = transformed.sel(coord_type=0), transformed.sel(coord_type=1)
    lon.name, lat.name = "lon", "lat"
    ds = ds.assign_coords(lat=lat, lon=lon).set_coords(["lat", "lon"])
    return ds

def ensure_cdo_compliance(ds, varname):
    for coord in ['lat', 'lon']:
        if coord in ds and coord not in ds.coords:
            ds = ds.set_coords(coord)
    ds[varname].attrs["coordinates"] = "lat lon"
    if "coordinates" in ds[varname].encoding:
        del ds[varname].encoding["coordinates"]
    if "N" in ds:
        ds["N"].attrs["units"] = "meters"
    if "E" in ds:
        ds["E"].attrs["units"] = "meters"
    return ds

def conservative_coarsening(ds, varname, block_size):
    da = ds[varname]
    if 'time' not in da.dims:
        da = da.expand_dims('time')
    lat, lon = ds['lat'], ds['lon']
    R = 6371000
    lat_rad = np.deg2rad(lat)
    dlat = np.deg2rad(np.diff(lat.mean('E')).mean().item())
    dlon = np.deg2rad(np.diff(lon.mean('N')).mean().item())
    area = (R**2) * dlat * dlon * np.cos(lat_rad)
    area = area.broadcast_like(da.isel(time=0)).expand_dims(time=da.sizes['time'])
    weighted = da.fillna(0) * area
    valid_area = area * da.notnull()
    coarsen_dims = {dim: block_size for dim in ['N', 'E'] if dim in da.dims}
    weighted_sum = weighted.coarsen(**coarsen_dims, boundary='trim').sum()
    area_sum = valid_area.coarsen(**coarsen_dims, boundary='trim').sum()
    data_coarse = (weighted_sum / area_sum).where(area_sum != 0)
    lat_coarse = lat.coarsen(N=block_size, boundary='trim').mean()
    lon_coarse = lon.coarsen(E=block_size, boundary='trim').mean()
    lon2d, lat2d = xr.broadcast(lon_coarse, lat_coarse)
    data_coarse = data_coarse.assign_coords(lat=lat2d, lon=lon2d)
    data_coarse.name = varname
    ds_out = data_coarse.to_dataset().set_coords(["lat", "lon"])
    return ds_out


#Commented out older function. 
"""def interp_xarray_cubic(coarse_ds, highres_ds, varname, out_path):
    lat_1d = coarse_ds['lat'][:, 0].values if coarse_ds['lat'].ndim == 2 else coarse_ds['lat'].values
    lon_1d = coarse_ds['lon'][0, :].values if coarse_ds['lon'].ndim == 2 else coarse_ds['lon'].values
    ds_lowres = coarse_ds.drop_vars([v for v in ['lat', 'lon'] if v in coarse_ds])
    ds_lowres = ds_lowres.rename({'N': 'lat', 'E': 'lon'})
    ds_lowres = ds_lowres.assign_coords(lat=lat_1d, lon=lon_1d)
    new_lat = highres_ds['lat']
    new_lon = highres_ds['lon']
    ds_lowres_filled = ds_lowres.fillna(-999)
    ds_interpolated = ds_lowres_filled.interp(
        lat=new_lat, lon=new_lon, method='cubic',
        kwargs={'bounds_error': False, 'fill_value': -999}
    )
    for v in ds_interpolated.data_vars:
        arr = ds_interpolated[v]
        arr = arr.where(~np.isclose(arr, -999, atol=1e-2), np.nan)
        mask = ~np.isnan(highres_ds[varname])
        arr = arr.where(mask)
        ds_interpolated[v] = arr
    ds_interpolated.to_netcdf(str(out_path), encoding={v: {"_FillValue": np.nan} for v in ds_interpolated.data_vars})
    ds_interpolated.close()"""

#New function

def interp_cdo_bicubic(coarse_ds, highres_ds, varname, out_path):
    # Save coarse and highres grids to temp files
    with tempfile.NamedTemporaryFile(suffix=".nc") as coarse_tmp, \
         tempfile.NamedTemporaryFile(suffix=".nc") as grid_tmp, \
         tempfile.NamedTemporaryFile(suffix=".nc") as interp_tmp:

        # Save coarse data
        coarse_ds.to_netcdf(coarse_tmp.name)
        # Save highres grid (only coordinates, minimal data)
        grid_ds = highres_ds[[varname]].isel(time=0, drop=True) if "time" in highres_ds.dims else highres_ds[[varname]]
        grid_ds.to_netcdf(grid_tmp.name)

        # CDO remapbic: remap to highres grid using bicubic interpolation
        cmd = [
            "cdo", f"remapbic,{grid_tmp.name}", coarse_tmp.name, interp_tmp.name
        ]
        subprocess.check_call(cmd)

        # Load result and save to out_path
        ds_interp = xr.open_dataset(interp_tmp.name)
        ds_interp.to_netcdf(str(out_path), encoding={v: {"_FillValue": np.nan} for v in ds_interp.data_vars})
        ds_interp.close()

def rename_to_standard(ds):
    rename_dict = {k: v for k, v in VAR_RENAME_MAP.items() if k in ds.data_vars}
    ds = ds.rename(rename_dict)
    return ds

def process_split(ds, varname, split_name, block_size=11):
    # Promote lat/lon if needed
    if 'lat' not in ds.coords or 'lon' not in ds.coords:
        ds = promote_latlon(ds, varname)
    ds = ds.sortby("time")
    step1_path = OUTPUT_DIR / f"{varname}_{split_name}_step1_latlon.nc"
    if step1_path.exists():
        ds = xr.open_dataset(step1_path)
    else:
        save(ds, step1_path)
    coarse_ds = conservative_coarsening(ds, varname, block_size=block_size)
    step2_path = OUTPUT_DIR / f"{varname}_{split_name}_step2_coarse.nc"
    if step2_path.exists():
        coarse_ds = xr.open_dataset(step2_path)
    else:
        save(coarse_ds, step2_path)
    step3_path = OUTPUT_DIR / f"{varname}_{split_name}_step3_interp.nc"
    if step3_path.exists():
        interp_ds = xr.open_dataset(step3_path)
    else:
        interp_xarray_cubic(coarse_ds, ds, varname, step3_path)
        interp_ds = xr.open_dataset(step3_path)
    return ds, interp_ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", type=str, required=True)
    args = parser.parse_args()
    varname = args.var

    dataset_map = {
        "precip": {
            "train": "precip_train_merged.nc",
            "val":   "precip_val_merged.nc",
            "test_file": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/RhiresD_1971_2023.nc",
            "varname_in_file": "precip",
            "scale_type": "minmax"
        },
        "temp": {
            "train": "temp_train_merged.nc",
            "val":   "temp_val_merged.nc",
            "test_file": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/TabsD_1971_2023.nc",
            "varname_in_file": "temp",
            "scale_type": "standard"
        },
        "tmin": {
            "train": "tmin_train_merged.nc",
            "val":   "tmin_val_merged.nc",
            "test_file": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/TminD_1971_2023.nc",
            "varname_in_file": "tmin",
            "scale_type": "standard"
        },
        "tmax": {
            "train": "tmax_train_merged.nc",
            "val":   "tmax_val_merged.nc",
            "test_file": "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/TmaxD_1971_2023.nc",
            "varname_in_file": "tmax",
            "scale_type": "standard"
        }
    }

    if varname not in dataset_map:
        raise ValueError(f"[ERROR] Unknown variable '{varname}'. Choose from {list(dataset_map.keys())}.")

    info = dataset_map[varname]
    infile_train = INPUT_DIR / info["train"]
    infile_val = INPUT_DIR / info["val"]
    infile_test = Path(info["test_file"])
    varname_in_file = info["varname_in_file"]
    scale_type = info["scale_type"]

    t0 = time.time()

    ds_train = xr.open_dataset(infile_train)
    ds_val = xr.open_dataset(infile_val)
    ds_test_full = xr.open_dataset(infile_test)
    ds_test_full = rename_to_standard(ds_test_full)
    ds_test = ds_test_full.sel(time=slice("2011-01-01", "2020-12-31"))

    # processing each split
    highres_train, upsampled_train = process_split(ds_train, varname_in_file, "train")
    highres_val, upsampled_val = process_split(ds_val, varname_in_file, "val")
    highres_test, upsampled_test = process_split(ds_test, varname_in_file, "test")

    y_train = highres_train[varname_in_file]

    # Only use train target for scaling params
    if scale_type == "minmax":
        stats = {
            "min": float(y_train.min().values),
            "max": float(y_train.max().values)
        }
    elif scale_type == "standard":
        stats = {
            "mean": float(y_train.mean().values),
            "std": float(y_train.std().values)
        }
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")

    def scale(arr, stats, scale_type):
        if scale_type == "standard":
            return (arr - stats['mean']) / stats['std']
        elif scale_type == "minmax":
            return (arr - stats['min']) / (stats['max'] - stats['min'])
        else:
            raise ValueError(f"Unknown scale_type: {scale_type}")

    for split, upsampled, highres in [
        ("train", upsampled_train, highres_train),
        ("val", upsampled_val, highres_val),
        ("test", upsampled_test, highres_test)
    ]:
        x_path = OUTPUT_DIR / f"combined_{varname}_input_{split}_chronological_scaled.nc"
        y_path = OUTPUT_DIR / f"combined_{varname}_target_{split}_chronological_scaled.nc"
        if not x_path.exists():
            x_scaled = scale(upsampled[varname_in_file], stats, scale_type)
            save(x_scaled.to_dataset(name=varname_in_file), x_path)
        if not y_path.exists():
            y_scaled = scale(highres[varname_in_file], stats, scale_type)
            save(y_scaled.to_dataset(name=varname_in_file), y_path)

    json_path = OUTPUT_DIR / f"combined_{varname}_scaling_params_chronological.json"
    if not json_path.exists():
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()