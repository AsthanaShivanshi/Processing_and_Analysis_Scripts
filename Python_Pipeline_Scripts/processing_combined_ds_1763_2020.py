import os
import json
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import subprocess
from pyproj import Transformer, datadir
import tempfile

np.random.seed(42)

proj_path = os.environ.get("PROJ_LIB") or "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvNew/share/proj"
os.environ["PROJ_LIB"] = proj_path
datadir.set_data_dir(proj_path)

CHUNK_DICT_RAW = {"time": 50, "E": 100, "N": 100}
CHUNK_DICT_LATLON = {"time": 50, "lat": 100, "lon": 100}

BASE_DIR = Path(os.environ["BASE_DIR"])
INPUT_DIR = BASE_DIR / "sasthana" / "Downscaling"/ "Processing_and_Analysis_Scripts" / "Combined_Dataset"
INPUT_DIR.mkdir(parents=True,exist_ok=True)
OUTPUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Combined_Chronological_Dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

def squeeze_latlon(ds):
    lat2d = ds['lat']
    lon2d = ds['lon']
    if 'time' in lat2d.dims:
        lat2d = lat2d.isel(time=0)
    if 'time' in lon2d.dims:
        lon2d = lon2d.isel(time=0)
    if 'time' in lat2d.coords:
        lat2d = lat2d.drop_vars('time')
    if 'time' in lon2d.coords:
        lon2d = lon2d.drop_vars('time')
    ds = ds.drop_vars(['lat', 'lon'])
    ds = ds.assign_coords(lat=lat2d, lon=lon2d)
    return ds

def conservative_coarsening(ds, varname, block_size):
    da = ds[varname]
    if 'time' not in da.dims and 'time' in ds.dims:
        da = da.expand_dims('time')
    lat, lon = ds['lat'], ds['lon']
    if 'time' in lat.dims:
        lat = lat.isel(time=0)
    if 'time' in lon.dims:
        lon = lon.isel(time=0)
    # Area calculation
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
    for arr_name in ['lat_coarse', 'lon_coarse']:
        arr = locals()[arr_name]
        if 'time' in arr.dims:
            arr = arr.isel(time=0)
        if 'time' in arr.coords:
            arr = arr.drop_vars('time')
        if arr_name == 'lat_coarse':
            lat_coarse = arr
        else:
            lon_coarse = arr
    # Broadcasting to 2D
    lat2d, lon2d = xr.broadcast(lat_coarse, lon_coarse)
    lat2d = lat2d.transpose('N', 'E')
    lon2d = lon2d.transpose('N', 'E')
    # Assign only lat/lon as coordinates for CDO
    data_coarse = data_coarse.assign_coords(lat=lat2d, lon=lon2d)
    data_coarse.name = varname
    ds_out = data_coarse.to_dataset().set_coords(["lat", "lon"])
    for v in list(ds_out.data_vars) + list(ds_out.coords):
        if hasattr(ds_out[v], "attrs") and "grid_mapping" in ds_out[v].attrs:
            del ds_out[v].attrs["grid_mapping"]
    if "grid_mapping" in ds_out.attrs:
        del ds_out.attrs["grid_mapping"]
    return ds_out

def interpolate_bicubic_shell(coarse_ds, target_ds, varname):
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, ds in zip(['coarse', 'target'], [coarse_ds, target_ds]):
            ds = squeeze_latlon(ds)
            for v in list(ds.coords):
                if v not in ['lat', 'lon', 'time']:
                    ds = ds.drop_vars(v)
            # Removing grid_mapping from everywhere
            for vv in list(ds.data_vars) + list(ds.coords):
                if hasattr(ds[vv], "attrs") and "grid_mapping" in ds[vv].attrs:
                    del ds[vv].attrs["grid_mapping"]
            if "grid_mapping" in ds.attrs:
                del ds.attrs["grid_mapping"]
            assert ds['lat'].dims == ('N', 'E'), "lat must be 2D with dims ('N', 'E')"
            assert ds['lon'].dims == ('N', 'E'), "lon must be 2D with dims ('N', 'E')"
            file_path = Path(tmpdir) / f"{name}.nc"
            # Keeping only var and lat/lon/time
            keep_vars = [varname]
            if 'time' in ds.coords:
                keep_vars.append('time')
            ds[keep_vars].transpose("time", "N", "E").to_netcdf(file_path)

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
        "pr": ("pr_merged.nc", "minmax", "pr"),
        "tas": ("tas_merged.nc", "standard", "tas"),
        "tasmin": ("tasmin_merged.nc", "standard", "tasmin"),
        "tasmax": ("tasmax_merged.nc", "standard", "tasmax"),
    }

    if varname not in dataset_map:
        raise ValueError(f"[ERROR] Unknown variable '{varname}'.")

    infile, scale_type, varname_in_file = dataset_map[varname]
    infile_path = INPUT_DIR / infile
    if not infile_path.exists():
        raise FileNotFoundError(f"[ERROR] File does not exist: {infile_path}")

    step1_path = OUTPUT_DIR / f"{varname}_step1_latlon.nc"
    highres_ds = xr.open_dataset(step1_path).chunk(get_chunk_dict(xr.open_dataset(step1_path)))
    if highres_ds['lat'].ndim == 3:
        highres_ds = squeeze_latlon(highres_ds)
    for v in ['lat1d', 'lon1d']:
        if v in highres_ds:
            highres_ds = highres_ds.drop_vars(v)
    if 'lat' in highres_ds.data_vars:
        highres_ds = highres_ds.set_coords('lat')
    if 'lon' in highres_ds.data_vars:
        highres_ds = highres_ds.set_coords('lon')
    for v in list(highres_ds.data_vars) + list(highres_ds.coords):
        if hasattr(highres_ds[v], "attrs") and "grid_mapping" in highres_ds[v].attrs:
            del highres_ds[v].attrs["grid_mapping"]
    if "grid_mapping" in highres_ds.attrs:
        del highres_ds.attrs["grid_mapping"]

    step2_path = OUTPUT_DIR / f"{varname}_step2_coarse.nc"
    if 'lat' not in highres_ds.coords or 'lon' not in highres_ds.coords:
        highres_ds = squeeze_latlon(highres_ds)
        highres_ds = promote_latlon(str(step1_path), varname_in_file)
        for v in list(highres_ds.data_vars) + list(highres_ds.coords):
            if hasattr(highres_ds[v], "attrs") and "grid_mapping" in highres_ds[v].attrs:
                del highres_ds[v].attrs["grid_mapping"]
        if "grid_mapping" in highres_ds.attrs:
            del highres_ds.attrs["grid_mapping"]
        highres_ds.to_netcdf(step1_path)

    if not step2_path.exists():
        coarse_ds = conservative_coarsening(highres_ds, varname_in_file, block_size=11)
        # only lat/lon as cords
        for v in list(coarse_ds.coords):
            if v not in ['lat', 'lon', 'time']:
                coarse_ds = coarse_ds.drop_vars(v)
        for v in list(coarse_ds.data_vars) + list(coarse_ds.coords):
            if hasattr(coarse_ds[v], "attrs") and "grid_mapping" in coarse_ds[v].attrs:
                del coarse_ds[v].attrs["grid_mapping"]
        if "grid_mapping" in coarse_ds.attrs:
            del coarse_ds.attrs["grid_mapping"]
        assert coarse_ds['lat'].dims == ('N', 'E')
        assert coarse_ds['lon'].dims == ('N', 'E')
        coarse_ds.to_netcdf(step2_path)

    coarse_ds = xr.open_dataset(step2_path).chunk(get_chunk_dict(xr.open_dataset(step2_path)))
    # lat/lon/time as coords
    for v in list(coarse_ds.coords):
        if v not in ['lat', 'lon', 'time']:
            coarse_ds = coarse_ds.drop_vars(v)
    for v in list(coarse_ds.data_vars) + list(coarse_ds.coords):
        if hasattr(coarse_ds[v], "attrs") and "grid_mapping" in coarse_ds[v].attrs:
            del coarse_ds[v].attrs["grid_mapping"]
    if "grid_mapping" in coarse_ds.attrs:
        del coarse_ds.attrs["grid_mapping"]
    assert coarse_ds['lat'].dims == ('N', 'E')
    assert coarse_ds['lon'].dims == ('N', 'E')

    step3_path = OUTPUT_DIR / f"{varname}_step3_interp.nc"
    if not step3_path.exists():
        interp_ds = interpolate_bicubic_shell(coarse_ds, highres_ds, varname_in_file)
        interp_ds = interp_ds.chunk(get_chunk_dict(interp_ds))
        interp_ds = squeeze_latlon(interp_ds)
        for v in list(interp_ds.coords):
            if v not in ['lat', 'lon', 'time']:
                interp_ds = interp_ds.drop_vars(v)
        for v in list(interp_ds.data_vars) + list(interp_ds.coords):
            if hasattr(interp_ds[v], "attrs") and "grid_mapping" in interp_ds[v].attrs:
                del interp_ds[v].attrs["grid_mapping"]
        if "grid_mapping" in interp_ds.attrs:
            del interp_ds.attrs["grid_mapping"]
        assert interp_ds['lat'].dims == ('N', 'E')
        assert interp_ds['lon'].dims == ('N', 'E')
        interp_ds.to_netcdf(step3_path)
        interp_ds.close()
    interp_ds = xr.open_dataset(step3_path).chunk(get_chunk_dict(xr.open_dataset(step3_path)))

    highres = highres_ds[varname_in_file].sel(time=slice("1771-01-01", "2020-12-31"))
    upsampled = interp_ds[varname_in_file].sel(time=slice("1771-01-01", "2020-12-31"))
    years = highres["time"].dt.year
    if not np.array_equal(highres["time"].values, upsampled["time"].values):
        raise ValueError("Time correspondence mismatch between highres and upsampled datasets.")

    train_mask = (years >= 1771) & (years <= 1980)
    val_mask = (years >= 1981) & (years <= 2010)
    test_mask = (years >= 2011) & (years <= 2020)

    x_train = upsampled.isel(time=train_mask)
    y_train = highres.isel(time=train_mask)
    x_val = upsampled.isel(time=val_mask)
    y_val = highres.isel(time=val_mask)
    x_test = upsampled.isel(time=test_mask)
    y_test = highres.isel(time=test_mask)

    with tempfile.NamedTemporaryFile(suffix=".nc") as tmpfile:
        y_train.to_netcdf(tmpfile.name)
        stats = get_cdo_stats(tmpfile.name, scale_type)

    x_train_scaled = apply_cdo_scaling(x_train, stats, scale_type)
    x_val_scaled = apply_cdo_scaling(x_val, stats, scale_type)
    y_train_scaled = apply_cdo_scaling(y_train, stats, scale_type)
    y_val_scaled = apply_cdo_scaling(y_val, stats, scale_type)
    x_test_scaled = apply_cdo_scaling(x_test, stats, scale_type)
    y_test_scaled = apply_cdo_scaling(y_test, stats, scale_type)

    x_train_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_combined_input_train_chronological_scaled.nc")
    y_train_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_combined_target_train_chronological_scaled.nc")
    x_val_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_combined_input_val_chronological_scaled.nc")
    y_val_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_combined_target_val_chronological_scaled.nc")
    x_test_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_combined_input_test_chronological_scaled.nc")
    y_test_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_combined_target_test_chronological_scaled.nc")

    with open(OUTPUT_DIR / f"{varname}_scaling_params_combined_chronological.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()