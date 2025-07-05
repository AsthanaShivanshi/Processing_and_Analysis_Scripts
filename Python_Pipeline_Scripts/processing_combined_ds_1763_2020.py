import os
import json
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import subprocess
import tempfile

BASE_DIR = Path(os.environ["BASE_DIR"])
INPUT_DIR = BASE_DIR / "sasthana" / "Downscaling"/ "Processing_and_Analysis_Scripts" / "Combined_Dataset"
OUTPUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Combined_Chronological_Dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_encoding(ds):
    for v in ds.data_vars:
        if "coordinates" in ds[v].encoding:
            del ds[v].encoding["coordinates"]
    return ds

def cdo_clean(ds, varname):
    # Ensure lat/lon are coordinates
    for coord in ['lat', 'lon']:
        if coord in ds and coord not in ds.coords:
            ds = ds.set_coords(coord)
    # Only keep main variable and all coordinates
    keep_vars = [varname] + list(ds.coords)
    ds = ds[keep_vars]
    # Set coordinates attribute for CDO
    ds[varname].attrs["coordinates"] = "lat lon"
    # Remove from encoding if present
    if "coordinates" in ds[varname].encoding:
        del ds[varname].encoding["coordinates"]
    return ds

def conservative_coarsening(ds, varname, block_size):
    da = ds[varname]
    lat, lon = ds['lat'], ds['lon']
    R = 6371000
    lat_rad = np.deg2rad(lat)
    dlat = np.deg2rad(np.diff(lat.mean('E')).mean().item())
    dlon = np.deg2rad(np.diff(lon.mean('N')).mean().item())
    area = (R**2) * dlat * dlon * np.cos(lat_rad)
    area = area.broadcast_like(da)
    weighted = da.fillna(0) * area
    valid_area = area * da.notnull()
    coarsen_dims = {dim: block_size for dim in ['N', 'E'] if dim in da.dims}
    weighted_sum = weighted.coarsen(**coarsen_dims, boundary='trim').sum()
    area_sum = valid_area.coarsen(**coarsen_dims, boundary='trim').sum()
    data_coarse = (weighted_sum / area_sum).where(area_sum != 0)
    lat_coarse = lat.mean('E').coarsen(N=block_size, boundary='trim').mean()
    lon_coarse = lon.mean('N').coarsen(E=block_size, boundary='trim').mean()
    lat2d, lon2d = xr.broadcast(lat_coarse, lon_coarse)
    lat2d = lat2d.transpose('N', 'E')
    lon2d = lon2d.transpose('N', 'E')
    data_coarse = data_coarse.assign_coords(lat=lat2d, lon=lon2d)
    data_coarse.name = varname
    ds_out = data_coarse.to_dataset()
    # Ensure lat/lon are coordinates
    for coord in ['lat', 'lon']:
        if coord in ds_out and coord not in ds_out.coords:
            ds_out = ds_out.set_coords(coord)
    return ds_out


def interpolate_bicubic_shell(coarse_ds, target_ds, varname):
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, ds in zip(['coarse', 'target'], [coarse_ds, target_ds]):
            ds = cdo_clean(ds, varname)
            file_path = Path(tmpdir) / f"{name}.nc"
            clean_encoding(ds).to_netcdf(file_path)
        output_file = Path(tmpdir) / "interp.nc"
        script_path = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "Python_Pipeline_Scripts" / "bicubic_interpolation.sh"
        subprocess.run([
            str(script_path), str(Path(tmpdir) / "coarse.nc"), str(Path(tmpdir) / "target.nc"), str(output_file)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", type=str, required=True)
    args = parser.parse_args()
    varname = args.var

    dataset_map = {
        "precip": ("pr_merged.nc", "minmax", "precip"),
        "temp": ("temp_merged.nc", "standard", "temp"),
        "tmin": ("tmin_merged.nc", "standard", "tmin"),
        "tmax": ("tmax_merged.nc", "standard", "tmax"),
    }

    if varname not in dataset_map:
        raise ValueError(f"Unknown variable '{varname}'.")

    infile, scale_type, varname_in_file = dataset_map[varname]
    infile_path = INPUT_DIR / infile
    if not infile_path.exists():
        raise FileNotFoundError(f"File not present: {infile_path}")

    ds = xr.open_dataset(infile_path)

    # Conservative coarsening
    coarse_ds = conservative_coarsening(ds, varname_in_file, block_size=11)
    clean_encoding(coarse_ds).to_netcdf(OUTPUT_DIR / f"{varname}_coarse.nc")

    # Bicubic interpolation
    interp_ds = interpolate_bicubic_shell(coarse_ds, ds, varname_in_file)
    clean_encoding(interp_ds).to_netcdf(OUTPUT_DIR / f"{varname}_interp.nc")

    # Chronological splits
    highres = ds[varname_in_file].sel(time=slice("1771-01-01", "2020-12-31"))
    upsampled = interp_ds[varname_in_file].sel(time=slice("1771-01-01", "2020-12-31"))
    years = highres["time"].dt.year
    if not np.array_equal(highres["time"].values, upsampled["time"].values):
        raise ValueError("Time correspondence mismatch")

    train_mask = (years >= 1771) & (years <= 1980)
    val_mask = (years >= 1981) & (years <= 2010)
    test_mask = (years >= 2011) & (years <= 2020)

    x_train = upsampled.isel(time=train_mask)
    y_train = highres.isel(time=train_mask)
    x_val = upsampled.isel(time=val_mask)
    y_val = highres.isel(time=val_mask)
    x_test = upsampled.isel(time=test_mask)
    y_test = highres.isel(time=test_mask)

    # Scaling via train params
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmpfile:
        clean_encoding(y_train).to_netcdf(tmpfile.name)
        stats = get_cdo_stats(tmpfile.name, scale_type)

    x_train_scaled = apply_cdo_scaling(x_train, stats, scale_type)
    y_train_scaled = apply_cdo_scaling(y_train, stats, scale_type)
    x_val_scaled = apply_cdo_scaling(x_val, stats, scale_type)
    y_val_scaled = apply_cdo_scaling(y_val, stats, scale_type)
    x_test_scaled = apply_cdo_scaling(x_test, stats, scale_type)
    y_test_scaled = apply_cdo_scaling(y_test, stats, scale_type)

    # Saving scaled splits
    clean_encoding(x_train_scaled).to_netcdf(OUTPUT_DIR / f"{varname}_input_train_chronological_scaled.nc")
    clean_encoding(y_train_scaled).to_netcdf(OUTPUT_DIR / f"{varname}_target_train_chronological_scaled.nc")
    clean_encoding(x_val_scaled).to_netcdf(OUTPUT_DIR / f"{varname}_input_val_chronological_scaled.nc")
    clean_encoding(y_val_scaled).to_netcdf(OUTPUT_DIR / f"{varname}_target_val_chronological_scaled.nc")
    clean_encoding(x_test_scaled).to_netcdf(OUTPUT_DIR / f"{varname}_input_test_chronological_scaled.nc")
    clean_encoding(y_test_scaled).to_netcdf(OUTPUT_DIR / f"{varname}_target_test_chronological_scaled.nc")

    # Saving params
    with open(OUTPUT_DIR / f"{varname}_scaling_params_combined_chronological.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("All output files written.")

if __name__ == "__main__":
    main()