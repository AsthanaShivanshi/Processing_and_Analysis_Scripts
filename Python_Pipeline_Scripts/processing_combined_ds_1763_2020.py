import os
import json
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import subprocess
import tempfile
from pyproj import Transformer, datadir
from dask.distributed import Client

BASE_DIR = Path(os.environ["BASE_DIR"])
INPUT_DIR = BASE_DIR / "sasthana" / "Downscaling"/ "Processing_and_Analysis_Scripts" / "Combined_Dataset"
OUTPUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Combined_Chronological_Dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
proj_path = os.environ.get("PROJ_LIB") or "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvNew/share/proj"
os.environ["PROJ_LIB"] = proj_path
datadir.set_data_dir(proj_path)

CHUNK_DICT_RAW = {"time": 100, "E": 100, "N": 100}
CHUNK_DICT_LATLON = {"time": 100, "lat": 100, "lon": 100}

def get_chunk_dict(ds):
    dims = set(ds.dims)
    if {"lat", "lon"}.issubset(dims):
        return CHUNK_DICT_LATLON
    elif {"E", "N"}.issubset(dims):
        return CHUNK_DICT_RAW
    else:
        raise ValueError(f"Dataset has unknown dimensions: {ds.dims}")

def promote_latlon(infile, varname):
    ds = xr.open_dataset(infile).chunk({"time": 100, "N": 100, "E": 100})
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

def interpolate_bicubic_shell(coarse_ds, target_ds, varname):
    with tempfile.TemporaryDirectory() as tmpdir:
        coarse_file = Path(tmpdir) / "coarse.nc"
        target_file = Path(tmpdir) / "target.nc"
        output_file = Path(tmpdir) / "interp.nc"
        ensure_cdo_compliance(coarse_ds, varname).to_netcdf(coarse_file)
        ensure_cdo_compliance(target_ds, varname).to_netcdf(target_file)
        script_path = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "Python_Pipeline_Scripts" / "bicubic_interpolation.sh"
        subprocess.run([
            str(script_path), str(coarse_file), str(target_file), str(output_file)
        ], check=True)
        return xr.open_dataset(output_file)[[varname]]

def get_cdo_stats(file_path, method):
    stats = {}
    if method == "standard":
        stats['mean'] = float(subprocess.check_output(["cdo", "output", "-fldmean", "-timmean", str(file_path)]).decode().strip())
        stats['std'] = float(subprocess.check_output(["cdo", "output", "-fldmean", "-timstd", str(file_path)]).decode().strip())
    elif method == "minmax":
        stats['min'] = float(subprocess.check_output(["cdo", "output", "-fldmin", "-timmin", str(file_path)]).decode().strip())
        stats['max'] = float(subprocess.check_output(["cdo", "output", "-fldmax", "-timmax", str(file_path)]).decode().strip())
    else:
        raise ValueError(f"Unsupported method: {method}")
    return stats

def apply_cdo_scaling(ds, stats, method):
    if method == "standard":
        return (ds - stats['mean']) / stats['std']
    elif method == "minmax":
        return (ds - stats['min']) / (stats['max'] - stats['min'])
    else:
        raise ValueError(f"Unknown method: {method}")

def save_netcdf(ds, varname, path):
    ds = ensure_cdo_compliance(ds, varname)
    ds.to_netcdf(
        path,
        engine="netcdf4",
        encoding={varname: {"zlib": True, "complevel": 4}}
    )
    ds.close()

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

    step1_path = OUTPUT_DIR / f"{varname}_step1_latlon.nc"
    if not step1_path.exists() or not {'lat', 'lon'}.issubset(xr.open_dataset(step1_path).coords):
        print(f"[INFO] Step 1: Preparing dataset for '{varname}'...")
        ds = xr.open_dataset(infile_path)
        ds = ds.chunk(get_chunk_dict(ds))
        if 'lat' in ds.coords and 'lon' in ds.coords:
            pass
        elif 'lat' in ds.data_vars and 'lon' in ds.data_vars:
            ds = ds.set_coords(['lat', 'lon'])
        else:
            ds.close()
            ds = promote_latlon(infile_path, varname_in_file)
        save_netcdf(ds, varname_in_file, step1_path)

    highres_ds = xr.open_dataset(step1_path).chunk(get_chunk_dict(xr.open_dataset(step1_path)))

    step2_path = OUTPUT_DIR / f"{varname}_step2_coarse.nc"
    if not step2_path.exists():
        coarse_ds = conservative_coarsening(highres_ds, varname_in_file, block_size=11)
        save_netcdf(coarse_ds, varname_in_file, step2_path)
    coarse_ds = xr.open_dataset(step2_path).chunk(get_chunk_dict(xr.open_dataset(step2_path)))

    step3_path = OUTPUT_DIR / f"{varname}_step3_interp.nc"
    if not step3_path.exists():
        interp_ds = interpolate_bicubic_shell(coarse_ds, highres_ds, varname_in_file)
        interp_ds = interp_ds.chunk(get_chunk_dict(interp_ds))
        save_netcdf(interp_ds, varname_in_file, step3_path)
    interp_ds = xr.open_dataset(step3_path).chunk(get_chunk_dict(xr.open_dataset(step3_path)))

    # Chron split: 1771–1980 train, 1981–2010 val, 2011–2020 test
    highres_ds = highres_ds.sortby("time")
    interp_ds = interp_ds.sortby("time")
    highres = highres_ds[varname_in_file].sel(time=slice("1771-01-01", "2020-12-31"))
    upsampled = interp_ds[varname_in_file].sel(time=slice("1771-01-01", "2020-12-31"))
    years = upsampled['time.year'].values

    train_mask = (years >= 1771) & (years <= 1980)
    val_mask   = (years >= 1981) & (years <= 2010)
    test_mask  = (years >= 2011) & (years <= 2020)

    x_train = upsampled.isel(time=train_mask)
    y_train = highres.isel(time=train_mask)
    x_val   = upsampled.isel(time=val_mask)
    y_val   = highres.isel(time=val_mask)
    x_test  = upsampled.isel(time=test_mask)
    y_test  = highres.isel(time=test_mask)

    # Scaling params
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmpfile:
        y_train.to_netcdf(tmpfile.name)
        stats = get_cdo_stats(tmpfile.name, scale_type)

    x_train_scaled = apply_cdo_scaling(x_train, stats, scale_type)
    x_val_scaled = apply_cdo_scaling(x_val, stats, scale_type)
    y_train_scaled = apply_cdo_scaling(y_train, stats, scale_type)
    y_val_scaled = apply_cdo_scaling(y_val, stats, scale_type)
    x_test_scaled = apply_cdo_scaling(x_test, stats, scale_type)
    y_test_scaled = apply_cdo_scaling(y_test, stats, scale_type)

    save_netcdf(x_train_scaled, varname_in_file, OUTPUT_DIR / f"combined_{varname}_input_train_chronological_scaled.nc")
    save_netcdf(y_train_scaled, varname_in_file, OUTPUT_DIR / f"combined_{varname}_target_train_chronological_scaled.nc")
    save_netcdf(x_val_scaled, varname_in_file, OUTPUT_DIR / f"combined_{varname}_input_val_chronological_scaled.nc")
    save_netcdf(y_val_scaled, varname_in_file, OUTPUT_DIR / f"combined_{varname}_target_val_chronological_scaled.nc")
    save_netcdf(x_test_scaled, varname_in_file, OUTPUT_DIR / f"combined_{varname}_input_test_chronological_scaled.nc")
    save_netcdf(y_test_scaled, varname_in_file, OUTPUT_DIR / f"combined_{varname}_target_test_chronological_scaled.nc")

    with open(OUTPUT_DIR / f"combined_{varname}_scaling_params_chronological.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Cleaning up intermed files (optional)
    # for step_path in [step1_path, step2_path, step3_path]:
    #     try:
    #         os.remove(step_path)
    #     except FileNotFoundError:
    #         pass

if __name__ == "__main__":
    client = Client(processes=False)
    try:
        main()
    finally:
        client.close()