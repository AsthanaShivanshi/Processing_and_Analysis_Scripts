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
from dask.diagnostics import ProgressBar
import psutil
from pyproj import datadir
os.environ["PROJ_LIB"] = datadir.get_data_dir()


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
    EE, NN = np.meshgrid(E, N, indexing="xy")

    transformer = Transformer.from_crs(
        crs_from="EPSG:2056",  # CH1903+ / LV95
        crs_to="EPSG:4326",    # WGS84 
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

    if not has_time:
        da = da.expand_dims('time')

    lat = ds[latname]
    lon = ds[lonname]

    # Computing area weights
    R = 6371000
    dlat = np.deg2rad(np.diff(lat[:, 0].values).mean())
    dlon = np.deg2rad(np.diff(lon[0, :].values).mean())
    area_np = (R ** 2) * dlat * dlon * np.cos(np.deg2rad(lat.values))
    area_da = xr.DataArray(area_np, dims=("N", "E"), coords={latname: lat, lonname: lon})

    for dim in da.dims:
        if dim not in area_da.dims:
            area_da = area_da.expand_dims({dim: da.sizes[dim]})

    weighted = da * area_da
    coarsen_dims = {da.dims[-2]: block_size, da.dims[-1]: block_size}
    weighted_sum = weighted.coarsen(**coarsen_dims, boundary="pad").sum()
    area_sum = area_da.coarsen(**coarsen_dims, boundary="pad").sum()
    data_coarse = weighted_sum / area_sum

    lat_coarse_1d = lat[:, 0].coarsen({lat.dims[0]: block_size}, boundary="pad").mean()
    lon_coarse_1d = lon[0, :].coarsen({lon.dims[1]: block_size}, boundary="pad").mean()
    lon_coarse_2d, lat_coarse_2d = np.meshgrid(lon_coarse_1d, lat_coarse_1d)

    data_coarse = data_coarse.assign_coords({
        'lat': (data_coarse.dims[-2:], lat_coarse_2d),
        'lon': (data_coarse.dims[-2:], lon_coarse_2d)
    })

    data_coarse.name = varname

    if not has_time:
        data_coarse = data_coarse.squeeze("time")

    ds_out = data_coarse.to_dataset()
    ds_out['lat'].attrs.update({'units': 'degrees_north', 'standard_name': 'latitude'})
    ds_out['lon'].attrs.update({'units': 'degrees_east', 'standard_name': 'longitude'})

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


def split(x, y, train_ratio, seed):
    np.random.seed(seed)
    indices = np.arange(x.sizes['time'])
    np.random.shuffle(indices)
    split_idx = int(train_ratio * len(indices))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    return (
        x.isel(time=train_idx),
        x.isel(time=val_idx),
        y.isel(time=train_idx),
        y.isel(time=val_idx)
    )


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
        raise ValueError(f"Variable '{varname}' not recognized.")

    infile, scale_type = dataset_map[varname]
    infile_path = INPUT_DIR / infile
    chunk_dict = {"time": 10, "N": 100, "E": 100}

    step1_path = OUTPUT_DIR / f"{varname}_step1_latlon.nc"
    if step1_path.exists():
        highres_ds = xr.open_dataset(step1_path).chunk(chunk_dict)
    else:
        print(f"[INFO] Promoting lat/lon for {varname}...")
        highres_ds = promote_latlon(infile_path, varname).chunk(chunk_dict)
        highres_ds.to_netcdf(step1_path)
        del highres_ds
        gc.collect()
        highres_ds = xr.open_dataset(step1_path).chunk(chunk_dict)

    step2_path = OUTPUT_DIR / f"{varname}_step2_coarse.nc"
    if step2_path.exists():
        coarse_ds = xr.open_dataset(step2_path).chunk(chunk_dict)
    else:
        print(f"[INFO] Coarsening {varname}...")
        coarse_ds = conservative_coarsening(highres_ds, varname, block_size=11).chunk(chunk_dict)
        coarse_ds.to_netcdf(step2_path)
        del coarse_ds
        gc.collect()
        coarse_ds = xr.open_dataset(step2_path).chunk(chunk_dict)

    step3_path = OUTPUT_DIR / f"{varname}_step3_interp.nc"
    if step3_path.exists():
        interp_ds = xr.open_dataset(step3_path).chunk(chunk_dict)
    else:
        print(f"[INFO] Interpolating {varname}...")
        interp_ds = interpolate_bicubic_ds(coarse_ds, highres_ds, varname).chunk(chunk_dict)
        interp_ds.to_netcdf(step3_path)
        del interp_ds
        gc.collect()
        interp_ds = xr.open_dataset(step3_path).chunk(chunk_dict)

    highres = highres_ds[varname]
    upsampled = interp_ds[varname].assign_coords({
        'lat': highres_ds['lat'],
        'lon': highres_ds['lon']
    })
    upsampled['lat'].attrs = highres_ds['lat'].attrs
    upsampled['lon'].attrs = highres_ds['lon'].attrs

    print(f"[INFO] Splitting into train/val for {varname}...")
    x_train, x_val, y_train, y_val = split(upsampled, highres, TRAIN_RATIO, SEED)

    print(f"[INFO] Scaling {varname}...")
    stats = get_cdo_stats(infile_path, scale_type)
    x_train_scaled = apply_cdo_scaling(x_train, stats, scale_type)
    x_val_scaled = apply_cdo_scaling(x_val, stats, scale_type)
    y_train_scaled = apply_cdo_scaling(y_train, stats, scale_type)
    y_val_scaled = apply_cdo_scaling(y_val, stats, scale_type)

    print(f"[INFO] Saving final output for {varname}...")
    x_train_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_input_train_scaled.nc")
    x_val_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_input_val_scaled.nc")
    y_train_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_target_train_scaled.nc")
    y_val_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_target_val_scaled.nc")
    with open(OUTPUT_DIR / f"{varname}_scaling_params.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[SUCCESS] Finished processing '{varname}'.")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    n_workers = 1
    threads_per_worker = 16
    mem_per_worker = "240GB"  # leaving buffer for memory


    print(f"[INFO] Launching Dask cluster: {n_workers} workers x {threads_per_worker} threads, {mem_per_worker} per worker")

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=mem_per_worker
    )
    client = Client(cluster)
    ProgressBar().register()

    try:
        main()
    finally:
        client.close()
        cluster.close()

        gc.collect()

print(f"[INFO] Memory usage at start: {psutil.virtual_memory().used / 1e9:.2f} GB")