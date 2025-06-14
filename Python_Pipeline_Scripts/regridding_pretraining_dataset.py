import os
import gc
import json
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import subprocess
from pyproj import Transformer, datadir
import dask.array as da
from dask.distributed import Client
from dask.diagnostics import ProgressBar

proj_path = os.environ.get("PROJ_LIB") or "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvNew/share/proj"
os.environ["PROJ_LIB"] = proj_path
datadir.set_data_dir(proj_path)
print(f"[DEBUG] PROJ_LIB set to: {proj_path}")

CHUNK_DICT = {"time": 50, "N": 100, "E": 100}
BASE_DIR = Path(os.environ["BASE_DIR"])
INPUT_DIR = BASE_DIR / "raw_data" / "Reconstruction_UniBern_1763_2020"
OUTPUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Pretraining_Dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.8
SEED = 42


def promote_latlon(infile, varname):
    ds = xr.open_dataset(infile)
    ds = ds.chunk(CHUNK_DICT)

    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

    def transform_coords(e, n):
        lon, lat = transformer.transform(e, n)
        return np.stack([lat, lon], axis=0)

    E = ds["E"]
    N = ds["N"]
    EE, NN = xr.broadcast(E, N)

    transformed = xr.apply_ufunc(
        transform_coords,
        EE, NN,
        input_core_dims=[["N", "E"], ["N", "E"]],
        output_core_dims=[["coord_type", "N", "E"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
    )

    lat = transformed.sel(coord_type=0)
    lon = transformed.sel(coord_type=1)

    lat.name = "lat"
    lon.name = "lon"

    ds = ds.assign_coords(lat=lat, lon=lon)
    ds = ds.set_coords(["lat", "lon"])

    ds.close()
    return ds


def conservative_coarsening(ds, varname, block_size, latname='lat', lonname='lon'):
    da = ds[varname]
    has_time = 'time' in da.dims

    if not has_time:
        da = da.expand_dims('time')

    lat = ds[latname]
    lon = ds[lonname]

    R = 6371000
    lat_vals = lat.isel(E=0).compute()
    lon_vals = lon.isel(N=0).compute()
    dlat = np.deg2rad(np.diff(lat_vals).mean())
    dlon = np.deg2rad(np.diff(lon_vals).mean())

    lat_rad = np.deg2rad(lat)
    area_da = (R ** 2) * dlat * dlon * np.cos(lat_rad)

    for dim in da.dims:
        if dim not in area_da.dims:
            area_da = area_da.broadcast_like(da)

    weighted = da * area_da
    coarsen_dims = {da.dims[-2]: block_size, da.dims[-1]: block_size}
    weighted_sum = weighted.coarsen(**coarsen_dims, boundary="pad").sum()
    area_sum = area_da.coarsen(**coarsen_dims, boundary="pad").sum()
    data_coarse = weighted_sum / area_sum

    lat_coarse_1d = lat.isel(E=0).coarsen(N=block_size, boundary="pad").mean().compute()
    lon_coarse_1d = lon.isel(N=0).coarsen(E=block_size, boundary="pad").mean().compute()
    lon_coarse_2d, lat_coarse_2d = np.meshgrid(lon_coarse_1d, lat_coarse_1d)

    data_coarse = data_coarse.assign_coords({
        'lat': (data_coarse.dims[-2:], lat_coarse_2d),
        'lon': (data_coarse.dims[-2:], lon_coarse_2d)
    })

    data_coarse.name = varname

    if not has_time:
        data_coarse = data_coarse.squeeze("time")

    ds.close()
    return data_coarse.to_dataset()


def interpolate_bicubic_ds(coarse_ds, target_ds, varname):
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        coarse_file = Path(tmpdir) / "coarse.nc"
        target_file = Path(tmpdir) / "target.nc"
        output_file = Path(tmpdir) / "interp.nc"

        coarse_ds.to_netcdf(coarse_file)
        target_ds.to_netcdf(target_file)

        subprocess.run(["cdo", f"remapbic,{target_file}", str(coarse_file), str(output_file)], check=True)

        interp_ds = xr.open_dataset(output_file)

    interp_ds = interp_ds.assign_coords(lat=target_ds['lat'], lon=target_ds['lon'])
    interp_ds['lat'].attrs = target_ds['lat'].attrs
    interp_ds['lon'].attrs = target_ds['lon'].attrs

    if varname not in interp_ds:
        interp_ds[varname] = interp_ds[list(interp_ds.data_vars)[0]]

    return interp_ds[[varname, 'lat', 'lon']]


def split(x, y, train_ratio, seed):
    np.random.seed(seed)
    indices = np.arange(x.sizes['time'])
    np.random.shuffle(indices)
    split_idx = int(train_ratio * len(indices))
    return (
        x.isel(time=indices[:split_idx]),
        x.isel(time=indices[split_idx:]),
        y.isel(time=indices[:split_idx]),
        y.isel(time=indices[split_idx:])
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
    parser.add_argument("--var", type=str, required=True)
    args = parser.parse_args()
    varname = args.var

    dataset_map = {
        "precip": ("precip_1771_2010.nc", "minmax"),
        "temp": ("temp_1771_2010.nc", "standard"),
        "tmin": ("tmin_1771_2010.nc", "standard"),
        "tmax": ("tmax_1771_2010.nc", "standard"),
    }

    infile, scale_type = dataset_map[varname]
    infile_path = INPUT_DIR / infile

    step1_path = OUTPUT_DIR / f"{varname}_step1_latlon.nc"
    if not step1_path.exists():
        print(f"[INFO] Step 1: Promoting lat/lon for {varname}...")
        ds = promote_latlon(infile_path, varname)
        ds.to_netcdf(step1_path)
        ds.close()
    else:
        print(f"[INFO] Step 1: Skipping (already exists)")

    highres_ds = xr.open_dataset(step1_path).chunk(CHUNK_DICT)

    step2_path = OUTPUT_DIR / f"{varname}_step2_coarse.nc"
    if not step2_path.exists():
        print(f"[INFO] Step 2: Coarsening {varname}...")
        coarse_ds = conservative_coarsening(highres_ds, varname, block_size=11)
        coarse_ds.to_netcdf(step2_path)
        coarse_ds.close()
    else:
        print(f"[INFO] Step 2: Skipping (already exists)")

    highres_ds = xr.open_dataset(step1_path).chunk(CHUNK_DICT)
    coarse_ds = xr.open_dataset(step2_path).chunk(CHUNK_DICT)

    step3_path = OUTPUT_DIR / f"{varname}_step3_interp.nc"
    if not step3_path.exists():
        print(f"[INFO] Step 3: Interpolating {varname}...")
        interp_ds = interpolate_bicubic_ds(coarse_ds, highres_ds, varname).chunk(CHUNK_DICT)
        interp_ds.to_netcdf(step3_path)
        interp_ds.close()
    else:
        print(f"[INFO] Step 3: Skipping (already exists)")

    interp_ds = xr.open_dataset(step3_path).chunk(CHUNK_DICT)

    print(f"[INFO] Step 4: Splitting and scaling {varname}...")
    highres = highres_ds[varname]
    upsampled = interp_ds[varname].assign_coords(lat=highres_ds['lat'], lon=highres_ds['lon'])
    upsampled['lat'].attrs = highres_ds['lat'].attrs
    upsampled['lon'].attrs = highres_ds['lon'].attrs

    x_train, x_val, y_train, y_val = split(upsampled, highres, TRAIN_RATIO, SEED)

    print(f"[INFO] Step 4: Computing scaling stats...")
    stats = get_cdo_stats(infile_path, scale_type)

    x_train_scaled = apply_cdo_scaling(x_train, stats, scale_type)
    x_val_scaled = apply_cdo_scaling(x_val, stats, scale_type)
    y_train_scaled = apply_cdo_scaling(y_train, stats, scale_type)
    y_val_scaled = apply_cdo_scaling(y_val, stats, scale_type)

    print(f"[INFO] Step 4: Saving final scaled datasets...")
    x_train_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_input_train_scaled.nc")
    x_val_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_input_val_scaled.nc")
    y_train_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_target_train_scaled.nc")
    y_val_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_target_val_scaled.nc")

    with open(OUTPUT_DIR / f"{varname}_scaling_params.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[SUCCESS] Finished processing '{varname}'.")


if __name__ == "__main__":
    print("[INFO] Starting Dask client...")
    client = Client(processes=False) 
    ProgressBar().register()

    try:
        main()
    finally:
        client.close()
