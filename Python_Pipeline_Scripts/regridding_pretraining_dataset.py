import os
import json
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import subprocess
from pyproj import Transformer, datadir
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import tempfile

# Set projection path
proj_path = os.environ.get("PROJ_LIB") or "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvNew/share/proj"
os.environ["PROJ_LIB"] = proj_path
datadir.set_data_dir(proj_path)

# Chunk configs to account for N/E or lat/lon retaned in the coarsened datasets
CHUNK_DICT_RAW = {"time": 50, "E": 100, "N": 100}
CHUNK_DICT_LATLON = {"time": 50, "lat": 100, "lon": 100}
 
BASE_DIR = Path(os.environ["BASE_DIR"])
INPUT_DIR = BASE_DIR / "raw_data" / "Reconstruction_UniBern_1763_2020"
OUTPUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Pretraining_Dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.8
SEED = 42

#Chunking based on configuration of the file
def get_chunk_dict(ds):
    dims = set(ds.dims)
    if {"lat", "lon"}.issubset(dims):
        return CHUNK_DICT_LATLON
    elif {"E", "N"}.issubset(dims):
        return CHUNK_DICT_RAW
    else:
        raise ValueError(f"Dataset has unknown dimensions: {ds.dims}")
    

def promote_latlon(infile, varname):
    ds = xr.open_dataset(infile).chunk({"time": 50, "N": 100, "E": 100})
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

        coarse_ds[[varname]].transpose("time", "N", "E").to_netcdf(coarse_file)
        target_ds[[varname]].transpose("time", "N", "E").to_netcdf(target_file)

        script_path = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "Python_Pipeline_Scripts" / "bicubic_interpolation.sh"

        subprocess.run([
            str(script_path), str(coarse_file), str(target_file), str(output_file)
        ], check=True)

        return xr.open_dataset(output_file)[[varname]]

def split_by_decade(x, y, val_ratio=0.2, seed=42):
    years = x['time'].dt.year.values
    decades = (years // 10) * 10
    unique_decades = np.sort(np.unique(decades))

    n_val_decades = max(1, int(val_ratio * len(unique_decades)))
    quarter = len(unique_decades) // 4
    mid_decades = unique_decades[quarter:-quarter]

    np.random.seed(seed)
    val_decades = np.random.choice(mid_decades, size=n_val_decades, replace=False)

    val_mask = np.isin(decades, val_decades)
    train_mask = ~val_mask

    return (
        x.isel(time=train_mask),
        x.isel(time=val_mask),
        y.isel(time=train_mask),
        y.isel(time=val_mask),
        sorted(val_decades.tolist())
    )

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", type=str, required=True)
    args = parser.parse_args()
    varname = args.var

    dataset_map = {
        "precip": ("precip_1771_2010.nc", "minmax", "precip"),
        "temp":   ("temp_1771_2010.nc", "standard", "temp"),
        "tmin":   ("tmin_1771_2010.nc", "standard", "tmin"),
        "tmax":   ("tmax_1771_2010.nc", "standard", "tmax"),
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
        ds.to_netcdf(step1_path)
        ds.close()

    highres_ds = xr.open_dataset(step1_path).chunk(get_chunk_dict(xr.open_dataset(step1_path)))

    step2_path = OUTPUT_DIR / f"{varname}_step2_coarse.nc"
    if not step2_path.exists():
        coarse_ds = conservative_coarsening(highres_ds, varname_in_file, block_size=11)
        coarse_ds.to_netcdf(step2_path)
        coarse_ds.close()

    coarse_ds = xr.open_dataset(step2_path).chunk(get_chunk_dict(xr.open_dataset(step2_path)))

    step3_path = OUTPUT_DIR / f"{varname}_step3_interp.nc"
    if not step3_path.exists():
        interp_ds = interpolate_bicubic_shell(coarse_ds, highres_ds, varname_in_file)
        interp_ds = interp_ds.chunk(get_chunk_dict(interp_ds))
        interp_ds.to_netcdf(step3_path)
        interp_ds.close()

    interp_ds = xr.open_dataset(step3_path).chunk(get_chunk_dict(xr.open_dataset(step3_path)))

    highres = highres_ds[varname_in_file]
    upsampled = interp_ds[varname_in_file]

    for coord in ['lat', 'lon']:
        if coord not in upsampled.coords:
            upsampled = upsampled.assign_coords({coord: highres_ds[coord]})
    upsampled['lat'].attrs = highres_ds['lat'].attrs
    upsampled['lon'].attrs = highres_ds['lon'].attrs

    x_train, x_val, y_train, y_val, val_decades = split_by_decade(
        upsampled, highres, val_ratio=0.2, seed=SEED
    )

    with open(OUTPUT_DIR / f"{varname}_val_decades.json", "w") as f:
        json.dump({"val_decades": val_decades}, f, indent=2)

    with tempfile.NamedTemporaryFile(suffix=".nc") as tmpfile:
        x_train.to_netcdf(tmpfile.name)
        stats = get_cdo_stats(tmpfile.name, scale_type)

    x_train_scaled = apply_cdo_scaling(x_train, stats, scale_type)
    x_val_scaled = apply_cdo_scaling(x_val, stats, scale_type)
    y_train_scaled = apply_cdo_scaling(y_train, stats, scale_type)
    y_val_scaled = apply_cdo_scaling(y_val, stats, scale_type)

    x_train_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_input_train_scaled.nc")
    x_val_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_input_val_scaled.nc")
    y_train_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_target_train_scaled.nc")
    y_val_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_target_val_scaled.nc")

    with open(OUTPUT_DIR / f"{varname}_scaling_params.json", "w") as f:
        json.dump(stats, f, indent=2)

    for step_path in [step1_path, step2_path, step3_path]:
        try:
            os.remove(step_path)
        except FileNotFoundError:
            pass

if __name__ == "__main__":
    client = Client(processes=False)
    ProgressBar().register()
    try:
        main()
    finally:
        client.close()
