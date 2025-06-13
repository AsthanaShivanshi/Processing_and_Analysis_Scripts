import xarray as xr
import numpy as np
import os
from pathlib import Path
import subprocess
import json
import gc
from dask.distributed import Client, LocalCluster
import argparse
from pyproj import Transformer
import psutil
import multiprocessing as mp

BASE_DIR = Path(os.environ["BASE_DIR"])
INPUT_DIR = BASE_DIR / "raw_data" / "Reconstruction_UniBern_1763_2020"
OUTPUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Pretraining_Dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.8
SEED = 42

def promote_latlon(infile, varname):
    ds = xr.open_dataset(infile)

    if not all(coord in ds.coords or coord in ds.dims for coord in ["E", "N"]):
        raise ValueError("Dataset must have 'E' and 'N' dimensions for projected coordinates.")

    E = ds["E"].values
    N = ds["N"].values
    EE, NN = np.meshgrid(E, N)

    transformer = Transformer.from_proj(
        proj_from="+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 "
                  "+k=1 +x_0=2600000 +y_0=1200000 +ellps=bessel +units=m +no_defs",
        proj_to="+proj=longlat +datum=WGS84 +no_defs",
        always_xy=True
    )

    lon, lat = transformer.transform(EE, NN)

    ds = ds.assign_coords(
        lat=(("N", "E"), lat),
        lon=(("N", "E"), lon)
    )
    ds = ds.set_coords(["lat", "lon"])
    return ds

def conservative_coarsening(ds, varname, block_size, latname='lat', lonname='lon'):
    da = ds[varname]
    has_time = 'time' in da.dims

    lat = ds[latname].values
    lon = ds[lonname].values

    ny, nx = lat.shape
    ny_pad = (block_size - ny % block_size) % block_size
    nx_pad = (block_size - nx % block_size) % block_size

    da = da.pad({da.dims[-2]: (0, ny_pad), da.dims[-1]: (0, nx_pad)}, mode='edge')
    lat = np.pad(lat, ((0, ny_pad), (0, nx_pad)), mode='edge')
    lon = np.pad(lon, ((0, ny_pad), (0, nx_pad)), mode='edge')

    R = 6371000
    dlat = np.deg2rad(np.diff(lat[:, 0]).mean())
    dlon = np.deg2rad(np.diff(lon[0, :]).mean())
    area = (R ** 2) * dlat * dlon * np.cos(np.deg2rad(lat))

    if not has_time:
        da = da.expand_dims('time')

    data = da.values
    area_blocks = area.reshape((area.shape[0] // block_size, block_size,
                                area.shape[1] // block_size, block_size))
    var_blocks = data.reshape((data.shape[0],
                               area.shape[0] // block_size, block_size,
                               area.shape[1] // block_size, block_size))

    weighted = (var_blocks * area_blocks).sum(axis=(2, 4))
    total_area = area_blocks.sum(axis=(1, 3))
    data_coarse = weighted / total_area

    lat_coarse = lat.reshape((lat.shape[0] // block_size, block_size,
                              lat.shape[1] // block_size, block_size)).mean(axis=(1, 3))
    lon_coarse = lon.reshape((lon.shape[0] // block_size, block_size,
                              lon.shape[1] // block_size, block_size)).mean(axis=(1, 3))

    coords = {
        "lat": (["y", "x"], lat_coarse),
        "lon": (["y", "x"], lon_coarse)
    }

    if has_time:
        coords["time"] = da["time"]
        dims = ("time", "y", "x")
    else:
        data_coarse = data_coarse.squeeze()
        dims = ("y", "x")

    var_da = xr.DataArray(data_coarse, coords=coords, dims=dims, name=varname)
    var_da.attrs = da.attrs

    ds_out = var_da.to_dataset()
    ds_out["lat"].attrs.update({'units': 'degrees_north', 'standard_name': 'latitude'})
    ds_out["lon"].attrs.update({'units': 'degrees_east', 'standard_name': 'longitude'})
    return ds_out

def interpolate_bicubic_ds(coarse_ds, target_ds, varname):
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        coarse_file = Path(tmpdir) / "coarse.nc"
        target_file = Path(tmpdir) / "target.nc"
        output_file = Path(tmpdir) / "interp.nc"

        coarse_ds.to_netcdf(coarse_file)
        target_ds.to_netcdf(target_file)

        cmd = ["cdo", f"remapbic,{target_file}", str(coarse_file), str(output_file)]
        subprocess.run(cmd, check=True)

        interp_ds = xr.open_dataset(output_file)

    # CDO output loses coords, so re-assign lat/lon from target_ds
    interp_ds = interp_ds.assign_coords(
        lat=target_ds['lat'],
        lon=target_ds['lon']
    )
    interp_ds['lat'].attrs = target_ds['lat'].attrs
    interp_ds['lon'].attrs = target_ds['lon'].attrs

    if varname not in interp_ds:
        interp_ds[varname] = interp_ds[list(interp_ds.data_vars)[0]]
    interp_ds = interp_ds[[varname, 'lat', 'lon']]

    return interp_ds

def split(ds, seed, train_ratio):
    np.random.seed(seed)
    indices = np.arange(ds.sizes['time'])
    np.random.shuffle(indices)
    split_idx = int(train_ratio * len(indices))
    return ds.isel(time=indices[:split_idx]), ds.isel(time=indices[split_idx:])

def get_cdo_stats(file_path, method):
    stats = {}
    file_path = str(file_path)

    if method == "standard":
        stats['mean'] = float(subprocess.check_output(["cdo", "output", "-fldmean", "-timmean", file_path]).decode().strip())
        stats['std'] = float(subprocess.check_output(["cdo", "output", "-fldmean", "-timstd", file_path]).decode().strip())

    elif method == "minmax":
        stats['min'] = float(subprocess.check_output(["cdo", "output", "-fldmin", "-timmin", file_path]).decode().strip())
        stats['max'] = float(subprocess.check_output(["cdo", "output", "-fldmax", "-timmax", file_path]).decode().strip())

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
    parser.add_argument("--var", type=str, required=True, help="Variable to process (e.g., precip, temp, tmin, tmax)")
    args = parser.parse_args()

    varname = args.var
    dataset_map = {
        "precip": ("precip_1771_2010.nc", "minmax"),
        "temp": ("temp_1771_2010.nc", "standard"),
        "tmin": ("tmin_1771_2010.nc", "standard"),
        "tmax": ("tmax_1771_2010.nc", "standard"),
    }

    if varname not in dataset_map:
        raise ValueError(f"Variable '{varname}' not recognized. Choose from: {list(dataset_map.keys())}")

    infile, scale_type = dataset_map[varname]
    infile_path = INPUT_DIR / infile

    # Promoting latlon in memory, using chunking
    highres_ds = promote_latlon(infile_path, varname).chunk({"time":100})

    # Conservative coarsening using chunking
    coarse_ds = conservative_coarsening(highres_ds, varname, block_size=11).chunk({"time":100})

    # Bicubic interpolate coarse to highres grid, in-memory
    interp_ds = interpolate_bicubic_ds(coarse_ds, highres_ds, varname).chunk({"time":100})

    highres = highres_ds[varname]
    upsampled = interp_ds[varname]

    # Assign coords to upsampled to match highres
    upsampled = upsampled.assign_coords({
        'lat': highres_ds['lat'],
        'lon': highres_ds['lon']
    })
    upsampled['lat'].attrs = highres_ds['lat'].attrs
    upsampled['lon'].attrs = highres_ds['lon'].attrs

    # Split train/val
    x_train, x_val = split(upsampled, SEED, TRAIN_RATIO)
    y_train, y_val = split(highres, SEED, TRAIN_RATIO)

    stats = get_cdo_stats(infile_path, scale_type)

    x_train_scaled = apply_cdo_scaling(x_train, stats, scale_type)
    x_val_scaled = apply_cdo_scaling(x_val, stats, scale_type)
    y_train_scaled = apply_cdo_scaling(y_train, stats, scale_type)
    y_val_scaled = apply_cdo_scaling(y_val, stats, scale_type)

    # Save only final scaled datasets
    x_train_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_input_train_scaled.nc")
    x_val_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_input_val_scaled.nc")
    y_train_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_target_train_scaled.nc")
    y_val_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_target_val_scaled.nc")

    with open(OUTPUT_DIR / f"{varname}_scaling_params.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Processed '{varname}' successfully.")

if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    total_mem_gb = psutil.virtual_memory().total / 1e9
    total_cores = psutil.cpu_count(logical=False)

    usable_mem_gb = int(total_mem_gb * 0.8)
    n_workers = max(1, min(16, total_cores // 2))
    mem_per_worker = f"{usable_mem_gb // n_workers}GB"

    print(f"[INFO] Total memory: {total_mem_gb:.1f} GB | Using {n_workers} workers @ {mem_per_worker} each")

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        memory_limit=mem_per_worker
    )
    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)

    main()
