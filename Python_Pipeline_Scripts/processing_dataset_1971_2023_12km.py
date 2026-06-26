import os
import json
import argparse
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import Transformer, datadir
from dask.distributed import Client

np.random.seed(42)

proj_path = os.environ.get("PROJ_LIB") or "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvNew/share/proj"
os.environ["PROJ_LIB"] = proj_path
datadir.set_data_dir(proj_path)

CHUNK_DICT_RAW = {"time": 50, "E": 100, "N": 100}
CHUNK_DICT_LATLON = {"time": 50, "lat": 100, "lon": 100}

BASE_DIR = Path(os.environ["BASE_DIR"])
INPUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "data_1971_2023" / "HR_files_full"
OUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Dataset_Setup_I_Chronological_12km"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "Python_Pipeline_Scripts"


def get_chunk_dict(ds: xr.Dataset):
    dims = set(ds.dims)
    if {"E", "N"}.issubset(dims):
        return CHUNK_DICT_RAW
    if {"lat", "lon"}.issubset(dims):
        return CHUNK_DICT_LATLON
    raise ValueError(f"Dataset has unknown dimensions: {ds.dims}")


def open_chunked(path: Path):
    ds = xr.open_dataset(path)
    return ds.chunk(get_chunk_dict(ds))


def promote_latlon(infile: Path):
    ds = xr.open_dataset(infile).chunk(CHUNK_DICT_RAW)
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

    def transform_coords(e, n):
        lon, lat = transformer.transform(e, n)
        return np.stack([lon, lat], axis=0)

    E, N = ds["E"], ds["N"]
    EE, NN = xr.broadcast(E, N)

    transformed = xr.apply_ufunc(
        transform_coords,
        EE,
        NN,
        input_core_dims=[["N", "E"], ["N", "E"]],
        output_core_dims=[["coord_type", "N", "E"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
    )

    lon = transformed.sel(coord_type=0).rename("lon")
    lat = transformed.sel(coord_type=1).rename("lat")
    return ds.assign_coords(lat=lat, lon=lon).set_coords(["lat", "lon"])


def conservative_coarsening(ds: xr.Dataset, varname: str, block_size: int):
    da = ds[varname]
    if "time" not in da.dims:
        da = da.expand_dims("time")

    lat = ds["lat"]
    lon = ds["lon"]

    R = 6371000.0
    lat_rad = np.deg2rad(lat)
    dlat = np.deg2rad(np.diff(lat.mean("E")).mean().item())
    dlon = np.deg2rad(np.diff(lon.mean("N")).mean().item())
    area = (R**2) * dlat * dlon * np.cos(lat_rad)

    area = area.broadcast_like(da.isel(time=0)).expand_dims(time=da.sizes["time"])
    weighted = da.fillna(0) * area
    valid_area = area * da.notnull()

    coarsen_dims = {dim: block_size for dim in ["N", "E"] if dim in da.dims}
    weighted_sum = weighted.coarsen(**coarsen_dims, boundary="trim").sum(skipna=True)
    area_sum = valid_area.coarsen(**coarsen_dims, boundary="trim").sum(skipna=True)
    data_coarse = (weighted_sum / area_sum).where(area_sum != 0)

    lat2d_coarse = lat.coarsen(N=block_size, E=block_size, boundary="trim").mean()
    lon2d_coarse = lon.coarsen(N=block_size, E=block_size, boundary="trim").mean()
    data_coarse = data_coarse.assign_coords(lat=lat2d_coarse, lon=lon2d_coarse)

    data_coarse.name = varname
    return data_coarse.to_dataset().set_coords(["lat", "lon"])


def run_cdo_interpolation(coarse_file: Path, target_file: Path, output_file: Path, method: str):
    if method not in {"bicubic", "bilinear"}:
        raise ValueError("method must be one of: bicubic, bilinear")

    script = "bicubic_interpolation.sh" if method == "bicubic" else "bilinear_interpolation.sh"
    script_path = SCRIPT_DIR / script
    os.chmod(script_path, 0o755)

    env = os.environ.copy()
    env["CDO_THREADS"] = env.get("CDO_THREADS", env.get("SLURM_CPUS_PER_TASK", "2"))

    subprocess.run(
        [str(script_path), str(coarse_file), str(target_file), str(output_file)],
        check=True,
        cwd=str(SCRIPT_DIR),
        env=env,
    )


def get_stats(da: xr.DataArray, method: str):
    stats = {}

    if method == "standard":
        stats["mean"] = float(da.mean(skipna=True).compute().item())
        stats["std"] = float(da.std(skipna=True).compute().item())

    elif method == "log":
        min_pos = da.where(da > 0).min(skipna=True).compute().item()
        epsilon = 1e-6 if (min_pos is None or np.isnan(min_pos) or min_pos <= 0) else float(min_pos) * 0.5
        log_da = np.log(da + epsilon)
        stats["epsilon"] = epsilon
        stats["mean"] = float(log_da.mean(skipna=True).compute().item())
        stats["std"] = float(log_da.std(skipna=True).compute().item())

    else:
        raise ValueError("Invalid scaling type. Available: standard, log")

    if stats["std"] == 0:
        raise ValueError("Scaling std is zero.")
    return stats


def apply_scaling(da: xr.DataArray, stats: dict, method: str):
    if method == "standard":
        return (da - stats["mean"]) / stats["std"]
    if method == "log":
        return (np.log(da + stats["epsilon"]) - stats["mean"]) / stats["std"]
    raise ValueError("Available: standard, log")


def save_split(
    x_train, y_train, x_val, y_val, x_test, y_test, stats, outdir: Path, varname: str, interp_method: str
):
    suffix = f"{varname}_{interp_method}"
    x_train.to_netcdf(outdir / f"{suffix}_input_train_scaled.nc")
    y_train.to_netcdf(outdir / f"{suffix}_target_train_scaled.nc")
    x_val.to_netcdf(outdir / f"{suffix}_input_val_scaled.nc")
    y_val.to_netcdf(outdir / f"{suffix}_target_val_scaled.nc")
    x_test.to_netcdf(outdir / f"{suffix}_input_test_scaled.nc")
    y_test.to_netcdf(outdir / f"{suffix}_target_test_scaled.nc")

    with open(outdir / f"{suffix}_scaling_params.json", "w") as f:
        json.dump(stats, f)


def final_outputs_exist(outdir: Path, varname: str, interp_method: str):
    suffix = f"{varname}_{interp_method}"
    files = [
        outdir / f"{suffix}_input_train_scaled.nc",
        outdir / f"{suffix}_target_train_scaled.nc",
        outdir / f"{suffix}_input_val_scaled.nc",
        outdir / f"{suffix}_target_val_scaled.nc",
        outdir / f"{suffix}_input_test_scaled.nc",
        outdir / f"{suffix}_target_test_scaled.nc",
        outdir / f"{suffix}_scaling_params.json",
    ]
    return all(f.exists() for f in files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", type=str, required=True, choices=["RhiresD", "TabsD"])
    args = parser.parse_args()
    varname = args.var

    dataset_map = {
        "RhiresD": ("RhiresD_1971_2023.nc", "log", "RhiresD"),
        "TabsD": ("TabsD_1971_2023.nc", "standard", "TabsD"),
    }

    infile, scale_type, varname_in_file = dataset_map[varname]
    infile_path = INPUT_DIR / infile

    step1_path = OUT_DIR / f"{varname}_step1_latlon.nc"
    needs_step1 = True
    if step1_path.exists():
        with xr.open_dataset(step1_path) as ds_chk:
            needs_step1 = not {"lat", "lon"}.issubset(ds_chk.coords)

    if needs_step1:
        print(f"[INFO] Step 1: preparing {varname}")
        ds = xr.open_dataset(infile_path).chunk(CHUNK_DICT_RAW)

        if "lat" in ds.coords and "lon" in ds.coords:
            pass
        elif "lat" in ds.data_vars and "lon" in ds.data_vars:
            ds = ds.set_coords(["lat", "lon"])
        else:
            ds.close()
            ds = promote_latlon(infile_path)

        if varname == "RhiresD":
            ds[varname_in_file] = xr.where(ds[varname_in_file] < 0.0, 0.0, ds[varname_in_file])

        ds.to_netcdf(step1_path)
        ds.close()

    highres_ds = open_chunked(step1_path)

    step2_path = OUT_DIR / f"{varname}_step2_coarse.nc"
    if not step2_path.exists():
        print(f"[INFO] Step 2: coarsening {varname}")
        coarse_ds_tmp = conservative_coarsening(highres_ds, varname_in_file, block_size=12)
        if varname == "RhiresD":
            coarse_ds_tmp[varname_in_file] = xr.where(coarse_ds_tmp[varname_in_file] < 0.0, 0.0, coarse_ds_tmp[varname_in_file])
        coarse_ds_tmp.to_netcdf(step2_path)
        coarse_ds_tmp.close()

    coarse_ds = open_chunked(step2_path)

    methods = ["bicubic", "bilinear"]
    step3_paths = {m: OUT_DIR / f"{varname}_step3_interp_{m}.nc" for m in methods}
    missing_step3 = [m for m in methods if not step3_paths[m].exists()]

    if missing_step3:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            coarse_file = tmpdir / "coarse.nc"
            target_file = tmpdir / "target_grid.nc"

            # Write remap inputs once
            coarse_ds[[varname_in_file]].transpose("time", "N", "E").to_netcdf(coarse_file)
            highres_ds[[varname_in_file]].isel(time=slice(0, 1)).transpose("time", "N", "E").to_netcdf(target_file)

            for interp_method in missing_step3:
                print(f"[INFO] Step 3: {interp_method} interpolation for {varname}")
                run_cdo_interpolation(coarse_file, target_file, step3_paths[interp_method], interp_method)

    # Prepare target + split + stats once
    highres = highres_ds[varname_in_file].sel(time=slice("1971-01-01", "2023-12-31"))
    if varname == "RhiresD":
        highres = xr.where(highres < 0.0, 0.0, highres)

    years = highres["time.year"].values
    train_mask = (years >= 1971) & (years <= 2004)
    val_mask = (years >= 2005) & (years <= 2014)
    test_mask = (years >= 2015) & (years <= 2023)

    y_train = highres.isel(time=train_mask)
    y_val = highres.isel(time=val_mask)
    y_test = highres.isel(time=test_mask)

    stats = get_stats(y_train, scale_type)
    y_train_scaled = apply_scaling(y_train, stats, scale_type)
    y_val_scaled = apply_scaling(y_val, stats, scale_type)
    y_test_scaled = apply_scaling(y_test, stats, scale_type)

    for interp_method in methods:
        if final_outputs_exist(OUT_DIR, varname, interp_method):
            print(f"[INFO] Skipping {varname}-{interp_method}: final files already exist")
            continue

        interp_ds = open_chunked(step3_paths[interp_method])
        upsampled = interp_ds[varname_in_file].sel(time=slice("1971-01-01", "2023-12-31"))

        if varname == "RhiresD":
            upsampled = xr.where(upsampled < 0.0, 0.0, upsampled)

        x_train = upsampled.isel(time=train_mask)
        x_val = upsampled.isel(time=val_mask)
        x_test = upsampled.isel(time=test_mask)

        x_train_scaled = apply_scaling(x_train, stats, scale_type)
        x_val_scaled = apply_scaling(x_val, stats, scale_type)
        x_test_scaled = apply_scaling(x_test, stats, scale_type)

        save_split(
            x_train_scaled, y_train_scaled,
            x_val_scaled, y_val_scaled,
            x_test_scaled, y_test_scaled,
            stats, OUT_DIR, varname, interp_method
        )

        interp_ds.close()

    highres_ds.close()
    coarse_ds.close()


if __name__ == "__main__":
    client = Client(processes=False)
    try:
        main()
    finally:
        client.close()