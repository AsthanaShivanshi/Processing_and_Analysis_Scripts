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
import subprocess

np.random.seed(42)

# For ensuring pyproj database directory is correctly set
proj_path = os.environ.get("PROJ_LIB") or "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvNew/share/proj"
os.environ["PROJ_LIB"] = proj_path
datadir.set_data_dir(proj_path)

CHUNK_DICT_RAW = {"time": 50, "E": 100, "N": 100}
CHUNK_DICT_LATLON = {"time": 50, "lat": 100, "lon": 100}
 
BASE_DIR = Path(os.environ["BASE_DIR"])
INPUT_DIR = BASE_DIR / "sasthana" / "Downscaling"/"Processing_and_Analysis_Scripts" / "data_1971_2023" / "HR_files_full"
OUTPUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Dataset_50km_SR_1971_2023"
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





def conservative_coarsening(ds, varname, block_size):  #Gives conservative coarsening depending on the block size
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




def coarsening_padding(ds, varname, pad_width=2):
    arr = ds[varname]
    arr_padded = arr.pad(
        N=(pad_width, pad_width), E=(pad_width, pad_width), 
        mode='edge')

    lat_padded = ds['lat'].pad(
        N=(pad_width, pad_width), E=(pad_width, pad_width), 
        mode='edge')
    lon_padded = ds['lon'].pad(
        N=(pad_width, pad_width), E=(pad_width, pad_width), 
        mode='edge')
    arr_padded = arr_padded.assign_coords(lat=lat_padded, lon=lon_padded)
    arr_padded.name = varname
    return arr_padded.to_dataset().set_coords(["lat", "lon"])




def interpolate_bicubic_shell(coarse_ds, target_ds, varname):

    n_size = coarse_ds.dims.get('N', None)
    e_size = coarse_ds.dims.get('E', None)
    pad = False
    if n_size is not None and e_size is not None:
        # Coarse grid: padding applied, else not
        if n_size < 10 and e_size < 10:
            pad = True
    with tempfile.TemporaryDirectory() as tmpdir:
        coarse_file = Path(tmpdir) / "coarse.nc"
        target_file = Path(tmpdir) / "target.nc"
        output_file = Path(tmpdir) / "interp.nc"
        if pad:
            coarse_to_write = coarsening_padding(coarse_ds, varname, pad_width=15)
        else:
            coarse_to_write = coarse_ds
        coarse_to_write[[varname]].transpose("time", "N", "E").to_netcdf(coarse_file)
        target_ds[[varname]].transpose("time", "N", "E").to_netcdf(target_file)
        script_path = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "Python_Pipeline_Scripts" / "bicubic_interpolation.sh"
        subprocess.run([
            str(script_path), str(coarse_file), str(target_file), str(output_file)
        ], check=True)
        return xr.open_dataset(output_file)[[varname]]


def get_cdo_stats(file_path, method,varname):
    stats = {}
    if method == "standard":
        stats['mean'] = float(subprocess.check_output(["cdo", "output", "-fldmean", "-timmean", str(file_path)]).decode().strip())
        stats['std'] = float(subprocess.check_output(["cdo", "output", "-fldmean", "-timstd", str(file_path)]).decode().strip())
    elif method == "minmax":
        stats['min'] = float(subprocess.check_output(["cdo", "output", "-fldmin", "-timmin", str(file_path)]).decode().strip())
        stats['max'] = float(subprocess.check_output(["cdo", "output", "-fldmax", "-timmax", str(file_path)]).decode().strip())
    elif method == "log":
        # mean and std dev after transform
        epsilon = 1e-3
        stats["epsilon"] = epsilon
        # Use cdo expr to log-transform and then standardise
        log_file = file_path + "_logtmp.nc"
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
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", type=str, required=True)
    args = parser.parse_args()
    varname = args.var

    dataset_map = {
        "RhiresD": ("RhiresD_1971_2023.nc", "log", "RhiresD"), #Changed from minmax here 
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

    step1_path = OUTPUT_DIR / f"{varname}_step1_latlon.nc"
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
        # Handling potential negative vals for RhiresD due to dirty data
        if varname == "RhiresD":
            ds[varname_in_file] = xr.where(ds[varname_in_file] < 0, 0, ds[varname_in_file])
        ds.to_netcdf(step1_path)
        ds.close()

    highres_ds = xr.open_dataset(step1_path).chunk(get_chunk_dict(xr.open_dataset(step1_path)))


    step2_path = OUTPUT_DIR / f"{varname}_step2_coarse.nc"
    if not step2_path.exists():
        coarse_ds = conservative_coarsening(highres_ds, varname_in_file, block_size=44) #EUR44
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



    step3_path = OUTPUT_DIR / f"{varname}_step3_interp.nc"
    if not step3_path.exists():
        interp_ds = interpolate_bicubic_shell(coarse_ds, highres_ds, varname_in_file)
        interp_ds = interp_ds.chunk(get_chunk_dict(interp_ds))
        interp_ds.to_netcdf(step3_path)
        interp_ds.close()
    interp_ds = xr.open_dataset(step3_path).chunk(get_chunk_dict(xr.open_dataset(step3_path)))

    # Chron split: 1971–2010 train, 2011–2020 val, 2021–2023 test
    highres = highres_ds[varname_in_file].sel(time=slice("1971-01-01", "2023-12-31"))
    upsampled = interp_ds[varname_in_file].sel(time=slice("1971-01-01", "2023-12-31"))
    years = upsampled['time.year'].values

    train_mask = (years >= 1971) & (years <= 2010)
    val_mask   = (years >= 2011) & (years <= 2020)
    test_mask  = (years >= 2021) & (years <= 2023)

    x_train = upsampled.isel(time=train_mask)
    y_train = highres.isel(time=train_mask)
    x_val   = upsampled.isel(time=val_mask)
    y_val   = highres.isel(time=val_mask)
    x_test  = upsampled.isel(time=test_mask)
    y_test  = highres.isel(time=test_mask)

    # Scaling params
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmpfile:
        y_train.to_netcdf(tmpfile.name)
        stats = get_cdo_stats(tmpfile.name, scale_type,varname_in_file)

    x_train_scaled = apply_cdo_scaling(x_train, stats, scale_type)
    x_val_scaled = apply_cdo_scaling(x_val, stats, scale_type)
    y_train_scaled = apply_cdo_scaling(y_train, stats, scale_type)
    y_val_scaled = apply_cdo_scaling(y_val, stats, scale_type)
    x_test_scaled = apply_cdo_scaling(x_test, stats, scale_type)
    y_test_scaled = apply_cdo_scaling(y_test, stats, scale_type)

    x_train_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_input_train_chronological_scaled.nc")
    y_train_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_target_train_chronological_scaled.nc")
    x_val_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_input_val_chronological_scaled.nc")
    y_val_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_target_val_chronological_scaled.nc")
    x_test_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_input_test_chronological_scaled.nc")
    y_test_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_target_test_chronological_scaled.nc")

    with open(OUTPUT_DIR / f"{varname}_scaling_params_chronological.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    client = Client(processes=False)
    try:
        main()
    finally:
        client.close()