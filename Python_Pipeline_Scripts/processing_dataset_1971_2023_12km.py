import os
import json
import argparse
import tempfile
import subprocess
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import xarray as xr
from pyproj import Transformer, datadir


np.random.seed(42)

proj_path = os.environ.get("PROJ_LIB") or "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvNew/share/proj"
os.environ["PROJ_LIB"] = proj_path
datadir.set_data_dir(proj_path)

BASE_DIR = Path(os.environ["BASE_DIR"])

INPUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "data_1971_2023" / "HR_files_full"
OUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Dataset_Setup_I_Chronological_12km"

OUT_DIR.mkdir(parents=True, exist_ok=True)

NETCDF_ENGINE = os.environ.get("XR_NETCDF_ENGINE", "h5netcdf")


def chunking(path):
    return xr.open_dataset(path).load()


def latlon(path):
    if not path.exists():
        return False
    with xr.open_dataset(path) as ds:
        return {"lat", "lon"}.issubset(ds.coords)


def netcdf(obj, path):
    obj.to_netcdf(path, engine=NETCDF_ENGINE)


def promote_latlon(infile, varname):
    ds = xr.open_dataset(infile).load()

    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

    e2d, n2d = xr.broadcast(ds["E"], ds["N"])
    lon_vals, lat_vals = transformer.transform(e2d.values, n2d.values)

    lon = xr.DataArray(
        lon_vals,
        dims=("N", "E"),
        coords={"N": ds["N"], "E": ds["E"]},
        name="lon",
    )

    lat = xr.DataArray(
        lat_vals,
        dims=("N", "E"),
        coords={"N": ds["N"], "E": ds["E"]},
        name="lat",
    )

    ds = ds.assign_coords(lat=lat, lon=lon).set_coords(["lat", "lon"])
    return ds[[varname]]


def conservative_coarsening(ds, varname, block_size):
    da = ds[varname]
    if "time" not in da.dims:
        da = da.expand_dims("time")

    lat = ds["lat"]
    lon = ds["lon"]
    r_earth = 6371000.0

    lat_rad = np.deg2rad(lat)
    dlat = np.deg2rad(np.diff(lat.mean("E").values).mean())
    dlon = np.deg2rad(np.diff(lon.mean("N").values).mean())
    area = (r_earth ** 2) * dlat * dlon * np.cos(lat_rad)

    area = area.broadcast_like(da.isel(time=0)).expand_dims(time=da["time"])
    weighted = da.fillna(0) * area
    valid_area = area * da.notnull()

    weighted_sum = weighted.coarsen(N=block_size, E=block_size, boundary="trim").sum()
    area_sum = valid_area.coarsen(N=block_size, E=block_size, boundary="trim").sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        data_coarse = xr.where(area_sum > 0, weighted_sum / area_sum, np.nan)

    lat_coarse = lat.coarsen(N=block_size, E=block_size, boundary="trim").mean()
    lon_coarse = lon.coarsen(N=block_size, E=block_size, boundary="trim").mean()

    data_coarse = data_coarse.assign_coords(lat=lat_coarse, lon=lon_coarse)
    data_coarse.name = varname

    return data_coarse.to_dataset().set_coords(["lat", "lon"])


def bilinear(coarse_ds, target_ds, varname, output_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        coarse_file = Path(tmpdir) / "coarse.nc"
        target_file = Path(tmpdir) / "target.nc"

        netcdf(coarse_ds[[varname]].transpose("time", "N", "E"), coarse_file)
        netcdf(target_ds[[varname]].transpose("time", "N", "E"), target_file)

        script_path = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "Python_Pipeline_Scripts" / "bilinear_interpolation.sh"
        working_dir = script_path.parent

        os.chmod(script_path, 0o755)

        subprocess.run(
            [str(script_path), str(coarse_file), str(target_file), str(output_file)],
            check=True,
            cwd=str(working_dir),
        )


def get_stats(da, method):
    stats = {}

    if method == "standard":
        stats["mean"] = float(da.mean(skipna=True).item())
        stats["std"] = float(da.std(skipna=True).item())

    elif method == "log":
        min_positive = da.where(da > 0).min(skipna=True).item()
        epsilon = float(min_positive) * 0.5

        log_da = np.log(da + epsilon)
        stats["epsilon"] = epsilon
        stats["mean"] = float(log_da.mean(skipna=True).item())
        stats["std"] = float(log_da.std(skipna=True).item())

    else:
        raise ValueError("Invalid scaling type")

    if not np.isfinite(stats["std"]) or stats["std"] == 0:
        raise ValueError("Scaling std must be finite and non-zero.")

    return stats


def apply_scaling(da, stats, method):
    if method == "standard":
        return (da - stats["mean"]) / stats["std"]

    if method == "log":
        log_da = np.log(da + stats["epsilon"])
        return (log_da - stats["mean"]) / stats["std"]

    raise ValueError("Available: standard, log")


def save_split(x, y, stats, outdir, varname, split_name, scale_type):
    outdir.mkdir(parents=True, exist_ok=True)

    x_scaled = apply_scaling(x, stats, scale_type)
    netcdf(
        x_scaled,
        outdir / f"{varname}_input_{split_name}_scaled.nc",
    )

    y_scaled = apply_scaling(y, stats, scale_type)
    netcdf(
        y_scaled,
        outdir / f"{varname}_target_{split_name}_scaled.nc",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", type=str, required=True, choices=["RhiresD", "TabsD", "TminD", "TmaxD"])
    args = parser.parse_args()
    varname = args.var

    dataset_map = {
        "RhiresD": ("RhiresD_1971_2023.nc", "log", "RhiresD"),
        "TabsD": ("TabsD_1971_2023.nc", "standard", "TabsD"),
        "TminD": ("TminD_1971_2023.nc", "standard", "TminD"),
        "TmaxD": ("TmaxD_1971_2023.nc", "standard", "TmaxD"),
    }

    infile, scale_type, varname_in_file = dataset_map[varname]
    infile_path = INPUT_DIR / infile

    step1_path = OUT_DIR / f"{varname}_step1_latlon.nc"
    step2_path = OUT_DIR / f"{varname}_step2_coarse.nc"
    step3_path = OUT_DIR / f"{varname}_step3_interp.nc"

    if not latlon(step1_path):
        print(f"Step 1: prepping '{varname}'")

        ds = xr.open_dataset(infile_path).load()

        if "lat" in ds.coords and "lon" in ds.coords:
            pass
        elif "lat" in ds.data_vars and "lon" in ds.data_vars:
            ds = ds.set_coords(["lat", "lon"])
        else:
            ds.close()
            ds = promote_latlon(infile_path, varname_in_file)

        if varname == "RhiresD":
            ds[varname_in_file] = xr.where(ds[varname_in_file] < 0, 0, ds[varname_in_file]) 

        netcdf(ds, step1_path)
        ds.close()

    highres_ds = chunking(step1_path)

    if not step2_path.exists():
        print(f"Step 2: coarsening '{varname}'")
        coarse_ds = conservative_coarsening(highres_ds, varname_in_file, block_size=12)

        if varname == "RhiresD":
            coarse_ds[varname_in_file] = xr.where(coarse_ds[varname_in_file] < 0, 0, coarse_ds[varname_in_file])

        netcdf(coarse_ds, step2_path)
        coarse_ds.close()

    coarse_ds = chunking(step2_path)

    if not step3_path.exists():
        print(f"Step 3: interpolating '{varname}'")
        bilinear(coarse_ds, highres_ds, varname_in_file, step3_path)

    interp_ds = chunking(step3_path)

    if varname == "RhiresD":
        interp_ds[varname_in_file] = xr.where(interp_ds[varname_in_file] < 0, 0, interp_ds[varname_in_file])

    highres = highres_ds[varname_in_file].sel(time=slice("1971-01-01", "2023-12-31"))
    upsampled = interp_ds[varname_in_file].sel(time=slice("1971-01-01", "2023-12-31"))

    upsampled, highres = xr.align(upsampled, highres, join="exact")

    x_train = upsampled.sel(time=slice("1971-01-01", "2004-12-31"))
    y_train = highres.sel(time=slice("1971-01-01", "2004-12-31"))

    x_val = upsampled.sel(time=slice("2005-01-01", "2014-12-31"))
    y_val = highres.sel(time=slice("2005-01-01", "2014-12-31"))

    x_test = upsampled.sel(time=slice("2015-01-01", "2023-12-31"))
    y_test = highres.sel(time=slice("2015-01-01", "2023-12-31"))

    stats = get_stats(y_train, scale_type)

    save_split(x_train, y_train, stats, OUT_DIR, varname, "train", scale_type)
    save_split(x_val, y_val, stats, OUT_DIR, varname, "val", scale_type)
    save_split(x_test, y_test, stats, OUT_DIR, varname, "test", scale_type)

    with open(OUT_DIR / f"{varname}_scaling_params.json", "w") as f:
        json.dump(stats, f)

    highres_ds.close()
    coarse_ds.close()
    interp_ds.close()


if __name__ == "__main__":
    main()