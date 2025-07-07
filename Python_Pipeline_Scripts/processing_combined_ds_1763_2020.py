import os
import json
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import tempfile
from pyproj import Transformer, datadir
import time

#No need for chunking with 500 GB RAM. only process one var per node 
#alternative, command line remapbic in cdo per file, not via a pipeline
#naN filling was performed simply because xarray interpolation function doesnt handle NaN or work with skipna
#this pipeline was made to interpolate without cdo because of the sheer size of the datasets, xarray interpolation with nan filling was performed as a result. Faster than OOM kill issues with CDO
BASE_DIR = Path(os.environ["BASE_DIR"])
INPUT_DIR = BASE_DIR / "sasthana" / "Downscaling"/ "Processing_and_Analysis_Scripts" / "Combined_Dataset"
OUTPUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Combined_Chronological_Dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
proj_path = os.environ.get("PROJ_LIB") or "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvNew/share/proj"
os.environ["PROJ_LIB"] = proj_path
datadir.set_data_dir(proj_path)

def get_chunk_dict(ds):
    return {}

def promote_latlon(infile, varname):
    ds = xr.open_dataset(infile)
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

def save(ds, path):
    encoding = {}
    for v in ds.data_vars:
        encoding[v] = {"_FillValue": np.nan}
    ds.to_netcdf(str(path), encoding=encoding)
    ds.close()

def interp_xarray_cubic(coarse_ds, highres_ds, varname, out_path):
    # Prepare 1D lat/lon from coarse grid
    lat_1d = coarse_ds['lat'][:, 0].values if coarse_ds['lat'].ndim == 2 else coarse_ds['lat'].values
    lon_1d = coarse_ds['lon'][0, :].values if coarse_ds['lon'].ndim == 2 else coarse_ds['lon'].values

    # Drop 2D lat/lon if present, assign 1D
    ds_lowres = coarse_ds.drop_vars([v for v in ['lat', 'lon'] if v in coarse_ds])
    ds_lowres = ds_lowres.rename({'N': 'lat', 'E': 'lon'})
    ds_lowres = ds_lowres.assign_coords(lat=lat_1d, lon=lon_1d)

    # Target grid
    new_lat = highres_ds['lat']
    new_lon = highres_ds['lon']

    # Fill NaNs with -999 before interpolation
    ds_lowres_filled = ds_lowres.fillna(-999)

    # Interpolate
    ds_interpolated = ds_lowres_filled.interp(
        lat=new_lat, lon=new_lon, method='cubic',
        kwargs={'bounds_error': False, 'fill_value': -999}
    )

    # Restore NaNs
    for v in ds_interpolated.data_vars:
        arr = ds_interpolated[v]
        arr = arr.where(~np.isclose(arr, -999, atol=1e-2), np.nan)

        # CLEANING STEP: Retain only valid grid cells from highres_ds for each timestep
        mask = ~np.isnan(highres_ds[varname])
        # Broadcast mask to arr shape if needed
        arr = arr.where(mask)

        ds_interpolated[v] = arr

    ds_interpolated.to_netcdf(str(out_path), encoding={v: {"_FillValue": np.nan} for v in ds_interpolated.data_vars})
    ds_interpolated.close()
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", type=str, required=True)
    args = parser.parse_args()
    varname = args.var

    dataset_map = {
        "precip": ("precip_merged.nc", "minmax", "precip"),
        "temp":   ("temp_merged.nc", "standard", "temp"),
        "tmin":   ("tmin_merged.nc", "standard", "tmin"),
        "tmax":   ("tmax_merged.nc", "standard", "tmax"),
    }

    if varname not in dataset_map:
        raise ValueError(f"[ERROR] Unknown variable '{varname}'. Choose from {list(dataset_map.keys())}.")

    infile, scale_type, varname_in_file = dataset_map[varname]
    infile_path = INPUT_DIR / infile
    if not infile_path.exists():
        raise FileNotFoundError(f"[ERROR] Input file does not exist: {infile_path}")

    t0 = time.time()
    step1_path = OUTPUT_DIR / f"{varname}_step1_latlon.nc"
    if not step1_path.exists():
        print(f"[INFO] Step 1: Preparing dataset for '{varname}'...")
        ds = xr.open_dataset(infile_path)
        if 'lat' in ds.coords and 'lon' in ds.coords:
            pass
        elif 'lat' in ds.data_vars and 'lon' in ds.data_vars:
            ds = ds.set_coords(['lat', 'lon'])
        else:
            ds.close()
            ds = promote_latlon(infile_path, varname_in_file)
        save(ds, step1_path)
    print(f"[TIMER] Step 1 done in {time.time() - t0:.1f} s")

    t1 = time.time()
    highres_ds = xr.open_dataset(step1_path)
    step2_path = OUTPUT_DIR / f"{varname}_step2_coarse.nc"
    if not step2_path.exists():
        coarse_ds = conservative_coarsening(highres_ds, varname_in_file, block_size=11)
        save(coarse_ds, step2_path)
    print(f"[TIMER] Step 2 done in {time.time() - t1:.1f} s")
    coarse_ds = xr.open_dataset(step2_path)

    t2 = time.time()
    step3_path = OUTPUT_DIR / f"{varname}_step3_interp.nc"
    if not step3_path.exists():
        interp_xarray_cubic(coarse_ds, highres_ds, varname_in_file, step3_path)
    print(f"[TIMER] Step 3 done in {time.time() - t2:.1f} s")
    interp_ds = xr.open_dataset(step3_path)

    # Chron split: 1771–1980 train, 1981–2010 val, 2011–2020 test
    highres_ds = highres_ds.sortby("time")
    interp_ds = interp_ds.sortby("time")
    highres = highres_ds[varname_in_file].sel(time=slice("1771-01-01", "2020-12-31"))
    upsampled = interp_ds[varname_in_file].sel(time=slice("1771-01-01", "2020-12-31"))

    x_train = upsampled.sel(time=slice("1771-01-01", "1980-12-31"))
    y_train = highres.sel(time=slice("1771-01-01", "1980-12-31"))
    x_val   = upsampled.sel(time=slice("1981-01-01", "2010-12-31"))
    y_val   = highres.sel(time=slice("1981-01-01", "2010-12-31"))
    x_test  = upsampled.sel(time=slice("2011-01-01", "2020-12-31"))
    y_test  = highres.sel(time=slice("2011-01-01", "2020-12-31"))

    # Scaling params
    stats = {}
    stats['mean'] = float(y_train.mean().values)
    stats['std'] = float(y_train.std().values)
    stats['min'] = float(y_train.min().values)
    stats['max'] = float(y_train.max().values)

    if scale_type == "standard":
        x_train_scaled = (x_train - stats['mean']) / stats['std']
        x_val_scaled = (x_val - stats['mean']) / stats['std']
        y_train_scaled = (y_train - stats['mean']) / stats['std']
        y_val_scaled = (y_val - stats['mean']) / stats['std']
        x_test_scaled = (x_test - stats['mean']) / stats['std']
        y_test_scaled = (y_test - stats['mean']) / stats['std']
    elif scale_type == "minmax":
        x_train_scaled = (x_train - stats['min']) / (stats['max'] - stats['min'])
        x_val_scaled = (x_val - stats['min']) / (stats['max'] - stats['min'])
        y_train_scaled = (y_train - stats['min']) / (stats['max'] - stats['min'])
        y_val_scaled = (y_val - stats['min']) / (stats['max'] - stats['min'])
        x_test_scaled = (x_test - stats['min']) / (stats['max'] - stats['min'])
        y_test_scaled = (y_test - stats['min']) / (stats['max'] - stats['min'])
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")

    save(x_train_scaled.to_dataset(name=varname_in_file), OUTPUT_DIR / f"combined_{varname}_input_train_chronological_scaled.nc")
    save(y_train_scaled.to_dataset(name=varname_in_file), OUTPUT_DIR / f"combined_{varname}_target_train_chronological_scaled.nc")
    save(x_val_scaled.to_dataset(name=varname_in_file), OUTPUT_DIR / f"combined_{varname}_input_val_chronological_scaled.nc")
    save(y_val_scaled.to_dataset(name=varname_in_file), OUTPUT_DIR / f"combined_{varname}_target_val_chronological_scaled.nc")
    save(x_test_scaled.to_dataset(name=varname_in_file), OUTPUT_DIR / f"combined_{varname}_input_test_chronological_scaled.nc")
    save(y_test_scaled.to_dataset(name=varname_in_file), OUTPUT_DIR / f"combined_{varname}_target_test_chronological_scaled.nc")

    with open(OUTPUT_DIR / f"combined_{varname}_scaling_params_chronological.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"All steps done in {time.time() - t0:.1f} s")

if __name__ == "__main__":
    main()