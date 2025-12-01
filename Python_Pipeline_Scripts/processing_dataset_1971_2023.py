import os
import json
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import subprocess
from pyproj import Transformer, datadir
from dask.distributed import Client
import tempfile

np.random.seed(42)

proj_path = os.environ.get("PROJ_LIB") or "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvNew/share/proj"
os.environ["PROJ_LIB"] = proj_path
datadir.set_data_dir(proj_path)

CHUNK_DICT_RAW = {"time": 50, "E": 100, "N": 100}
CHUNK_DICT_LATLON = {"time": 50, "lat": 100, "lon": 100}

BASE_DIR = Path(os.environ["BASE_DIR"])
INPUT_DIR = BASE_DIR / "sasthana" / "Downscaling"/"Processing_and_Analysis_Scripts" / "data_1971_2023" / "HR_files_full"

FOLD_I_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset_Fold_I"
FOLD_II_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Chronological_Dataset_Fold_II"
FOLD_I_DIR.mkdir(parents=True, exist_ok=True)
FOLD_II_DIR.mkdir(parents=True, exist_ok=True)




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
    lat2d_coarse = lat.coarsen(N=block_size, E=block_size, boundary='trim').mean()
    lon2d_coarse = lon.coarsen(N=block_size, E=block_size, boundary='trim').mean()
    data_coarse = data_coarse.assign_coords(lat=lat2d_coarse, lon=lon2d_coarse)
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



def get_cdo_stats(file_path, method, varname):
    stats = {}
    if method == "standard":
        stats['mean'] = float(subprocess.check_output(["cdo", "output", "-fldmean", "-timmean", str(file_path)]).decode().strip())
        stats['std'] = float(subprocess.check_output(["cdo", "output", "-fldmean", "-timstd", str(file_path)]).decode().strip())
    elif method == "minmax":
        stats['min'] = float(subprocess.check_output(["cdo", "output", "-fldmin", "-timmin", str(file_path)]).decode().strip())
        stats['max'] = float(subprocess.check_output(["cdo", "output", "-fldmax", "-timmax", str(file_path)]).decode().strip())
    elif method == "log":
        epsilon = 1e-3
        stats["epsilon"] = epsilon
        log_file = str(file_path) + "_logtmp.nc"
        subprocess.run(["cdo", f"expr,{varname}=log({varname}+{epsilon})", str(file_path), log_file], check=True)
        stats['mean'] = float(subprocess.check_output(["cdo", "output", "-fldmean", "-timmean", log_file]).decode().strip())
        stats['std'] = float(subprocess.check_output(["cdo", "output", "-fldmean", "-timstd", log_file]).decode().strip())
        os.remove(log_file)
    else:
        raise ValueError(f"Unsupported method: {method}")
    return stats



def apply_cdo_scaling(ds, stats, method):
    if method == "standard":
        return (ds - stats['mean']) / stats['std']
    elif method == "minmax":
        return (ds - stats['min']) / (stats['max'] - stats['min'])
    elif method == "log":
        log_ds = np.log(ds + stats["epsilon"])
        return (log_ds - stats['mean']) / stats['std']
    else:
        raise ValueError(f"Unknown method: {method}")



def save_fold(x_train, y_train, x_val, y_val, x_test, y_test, stats, outdir, varname, fold_name):
    x_train.to_netcdf(outdir / f"{varname}_input_train_{fold_name}_scaled.nc")
    y_train.to_netcdf(outdir / f"{varname}_target_train_{fold_name}_scaled.nc")
    x_val.to_netcdf(outdir / f"{varname}_input_val_{fold_name}_scaled.nc")
    y_val.to_netcdf(outdir / f"{varname}_target_val_{fold_name}_scaled.nc")
    x_test.to_netcdf(outdir / f"{varname}_input_test_{fold_name}_scaled.nc")
    y_test.to_netcdf(outdir / f"{varname}_target_test_{fold_name}_scaled.nc")
    with open(outdir / f"{varname}_scaling_params_{fold_name}.json", "w") as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", type=str, required=True)
    args = parser.parse_args()
    varname = args.var

    dataset_map = {
        "RhiresD": ("RhiresD_1971_2023_nonneg.nc", "log", "RhiresD"),
        "TabsD":   ("TabsD_1971_2023.nc", "standard", "TabsD"),
        "TminD":   ("TminD_1971_2023.nc", "standard", "TminD"),
        "TmaxD":   ("TmaxD_1971_2023.nc", "standard", "TmaxD"),
    }

    if varname not in dataset_map:
        raise ValueError(f"[ERROR] Unknown variable '{varname}'. Choose from {list(dataset_map.keys())}.")

    infile, scale_type, varname_in_file = dataset_map[varname]
    infile_path = INPUT_DIR / infile
    if not infile_path.exists():
        raise FileNotFoundError(f"[ERROR] Input file does not exist: {infile_path}")

    step1_path = FOLD_I_DIR / f"{varname}_step1_latlon.nc"
    if not step1_path.exists() or not {'lat', 'lon'}.issubset(xr.open_dataset(step1_path).coords):
        print(f"[INFO] Step 1: Preparing dataset for '{varname}'")
        ds = xr.open_dataset(infile_path)
        ds = ds.chunk(get_chunk_dict(ds))
        if 'lat' in ds.coords and 'lon' in ds.coords:
            pass
        elif 'lat' in ds.data_vars and 'lon' in ds.data_vars:
            ds = ds.set_coords(['lat', 'lon'])
        else:
            ds.close()
            ds = promote_latlon(infile_path, varname_in_file)
        if varname == "RhiresD":
            ds[varname_in_file] = xr.where(ds[varname_in_file] < 0, 0, ds[varname_in_file])
        ds.to_netcdf(step1_path)
        ds.close()

    highres_ds = xr.open_dataset(step1_path).chunk(get_chunk_dict(xr.open_dataset(step1_path)))

    step2_path = FOLD_I_DIR / f"{varname}_step2_coarse.nc"
    if not step2_path.exists():
        coarse_ds = conservative_coarsening(highres_ds, varname_in_file, block_size=11)
        coarse_ds.to_netcdf(step2_path)
        coarse_ds.close()
    coarse_ds = xr.open_dataset(step2_path).chunk(get_chunk_dict(xr.open_dataset(step2_path)))

    step3_path = FOLD_I_DIR / f"{varname}_step3_interp.nc"
    if not step3_path.exists():
        interp_ds = interpolate_bicubic_shell(coarse_ds, highres_ds, varname_in_file)
        interp_ds = interp_ds.chunk(get_chunk_dict(interp_ds))
        interp_ds.to_netcdf(step3_path)
        interp_ds.close()
    interp_ds = xr.open_dataset(step3_path).chunk(get_chunk_dict(xr.open_dataset(step3_path)))

    highres = highres_ds[varname_in_file].sel(time=slice("1971-01-01", "2023-12-31"))
    upsampled = interp_ds[varname_in_file].sel(time=slice("1971-01-01", "2023-12-31"))
    years = upsampled['time.year'].values

    # FOLD I: Train 1971–1995, Validate 1996–2020, Test 2021–2023
    train_mask_I = (years >= 1971) & (years <= 1995)
    val_mask_I   = (years >= 1996) & (years <= 2020)
    test_mask_I  = (years >= 2021) & (years <= 2023)
    x_train_I = upsampled.isel(time=train_mask_I)
    y_train_I = highres.isel(time=train_mask_I)
    x_val_I   = upsampled.isel(time=val_mask_I)
    y_val_I   = highres.isel(time=val_mask_I)
    x_test_I  = upsampled.isel(time=test_mask_I)
    y_test_I  = highres.isel(time=test_mask_I)
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmpfile_I:
        y_train_I.to_netcdf(tmpfile_I.name)
        stats_I = get_cdo_stats(tmpfile_I.name, scale_type, varname_in_file)
    x_train_I_scaled = apply_cdo_scaling(x_train_I, stats_I, scale_type)
    x_val_I_scaled = apply_cdo_scaling(x_val_I, stats_I, scale_type)
    x_test_I_scaled = apply_cdo_scaling(x_test_I, stats_I, scale_type)
    y_train_I_scaled = apply_cdo_scaling(y_train_I, stats_I, scale_type)
    y_val_I_scaled = apply_cdo_scaling(y_val_I, stats_I, scale_type)
    y_test_I_scaled = apply_cdo_scaling(y_test_I, stats_I, scale_type)
    save_fold(x_train_I_scaled, y_train_I_scaled, x_val_I_scaled, y_val_I_scaled, x_test_I_scaled, y_test_I_scaled, stats_I, FOLD_I_DIR, varname, "fold_I")

    # FOLD II: Train 1996–2020, Validate 1971–1995, Test 2021–2023
    train_mask_II = (years >= 1996) & (years <= 2020)
    val_mask_II   = (years >= 1971) & (years <= 1995)
    test_mask_II  = (years >= 2021) & (years <= 2023)
    x_train_II = upsampled.isel(time=train_mask_II)
    y_train_II = highres.isel(time=train_mask_II)
    x_val_II   = upsampled.isel(time=val_mask_II)
    y_val_II   = highres.isel(time=val_mask_II)
    x_test_II  = upsampled.isel(time=test_mask_II)
    y_test_II  = highres.isel(time=test_mask_II)
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmpfile_II:
        y_train_II.to_netcdf(tmpfile_II.name)
        stats_II = get_cdo_stats(tmpfile_II.name, scale_type, varname_in_file)
    x_train_II_scaled = apply_cdo_scaling(x_train_II, stats_II, scale_type)
    x_val_II_scaled = apply_cdo_scaling(x_val_II, stats_II, scale_type)
    x_test_II_scaled = apply_cdo_scaling(x_test_II, stats_II, scale_type)
    y_train_II_scaled = apply_cdo_scaling(y_train_II, stats_II, scale_type)
    y_val_II_scaled = apply_cdo_scaling(y_val_II, stats_II, scale_type)
    y_test_II_scaled = apply_cdo_scaling(y_test_II, stats_II, scale_type)
    save_fold(x_train_II_scaled, y_train_II_scaled, x_val_II_scaled, y_val_II_scaled, x_test_II_scaled, y_test_II_scaled, stats_II, FOLD_II_DIR, varname, "fold_II")



if __name__ == "__main__":
    client = Client(processes=False)
    try:
        main()
    finally:
        client.close()