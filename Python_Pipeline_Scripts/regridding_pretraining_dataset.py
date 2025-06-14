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

CHUNK_DICT_RAW = {"time": 50, "E": 100, "N": 100}  # used before coarsening
CHUNK_DICT_LATLON = {"time": 50, "lat": 100, "lon": 100}  # used after coarsening

BASE_DIR = Path(os.environ["BASE_DIR"])
INPUT_DIR = BASE_DIR / "raw_data" / "Reconstruction_UniBern_1763_2020"
OUTPUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Pretraining_Dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.8
SEED = 42


def promote_latlon(infile, varname):
    ds = xr.open_dataset(infile)
    ds = ds.chunk({"time": 50, "N": 100, "E": 100})

    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

    def transform_coords(e, n):
        lon, lat = transformer.transform(e, n)
        return np.stack([lon, lat], axis=0)  

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

    lon = transformed.sel(coord_type=0)
    lat = transformed.sel(coord_type=1)

    lon.name = "lon"
    lat.name = "lat"

    ds = ds.assign_coords(lat=lat, lon=lon)
    ds = ds.set_coords(["lat", "lon"])

    return ds


def conservative_coarsening(ds, varname, block_size):
    da = ds[varname]
    has_time = 'time' in da.dims
    if not has_time:
        da = da.expand_dims('time')

    lat = ds['lat']
    lon = ds['lon']

    R = 6371000
    lat_rad = np.deg2rad(lat)
    dlat = np.deg2rad(np.diff(lat.mean('E')).mean().item())
    dlon = np.deg2rad(np.diff(lon.mean('N')).mean().item())
    area = (R**2) * dlat * dlon * np.cos(lat_rad)

    # Ensure area is same shape as data
    area = area.broadcast_like(da.isel(time=0))
    area = area.expand_dims(time=da.sizes['time'])

    # Weighting and masking
    weighted = da.fillna(0) * area
    valid_area = area * da.notnull()

    coarsen_dims = {dim: block_size for dim in ['N', 'E'] if dim in da.dims}
    weighted_sum = weighted.coarsen(**coarsen_dims, boundary='trim').sum()
    area_sum = valid_area.coarsen(**coarsen_dims, boundary='trim').sum()
    data_coarse = (weighted_sum / area_sum).where(area_sum != 0)

    # Coarsen lat/lon (only 4 mtadata)
    lat_coarse = lat.coarsen(N=block_size, boundary='trim').mean()
    lon_coarse = lon.coarsen(E=block_size, boundary='trim').mean()
    lon2d, lat2d = xr.broadcast(lon_coarse, lat_coarse)

    data_coarse= data_coarse.assign_coords(lat=lat2d, lon=lon2d)
    data_coarse.name=varname
    
    ds_out=data_coarse.to_dataset()
    ds_out=ds_out.set_coords(["lat","lon"])
    return ds_out


def interpolate_bicubic_ds(coarse_ds, target_ds, varname):
    import tempfile

    def to_latlon_dims(ds, varname):
        """Temporarily rename Y/X dims to lat/lon dims for CDO compatibility."""
        da = ds[varname]
        da_latlon = da.rename({"Y": "lat", "X": "lon"})

        lat = ds["lat"].rename({"Y": "lat", "X": "lon"})
        lon = ds["lon"].rename({"Y": "lat", "X": "lon"})

        return da_latlon.to_dataset(name=varname).assign_coords(lat=lat, lon=lon)

    with tempfile.TemporaryDirectory() as tmpdir:
        coarse_file = Path(tmpdir) / "coarse.nc"
        target_file = Path(tmpdir) / "target.nc"
        output_file = Path(tmpdir) / "interp.nc"

        # Convert to lat/lon (CDO requirement)
        coarse_for_cdo = to_latlon_dims(coarse_ds, varname)
        target_for_cdo = to_latlon_dims(target_ds, varname)

        coarse_for_cdo.to_netcdf(coarse_file)
        target_for_cdo.to_netcdf(target_file)

        subprocess.run(["cdo", f"remapbic,{target_file}", str(coarse_file), str(output_file)], check=True)
        interp_ds = xr.open_dataset(output_file)

    interp_da = interp_ds[varname].rename({"lat": "Y", "lon": "X"})

    lat = target_ds["lat"]
    lon = target_ds["lon"]
    interp_da = interp_da.assign_coords(lat=lat, lon=lon)
    interp_da.name = varname

    return interp_da.to_dataset()

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

    # 1. Promoting lat/lon coordinates
    step1_path = OUTPUT_DIR / f"{varname}_step1_latlon.nc"
    needs_recompute = False

    if not step1_path.exists():
        needs_recompute = True
    else:
        with xr.open_dataset(step1_path) as ds_check:
            if 'lat' not in ds_check.coords or 'lon' not in ds_check.coords:
                print(f"[WARN] step1_path exists but is invalid. Recomputing...")
                needs_recompute = True

    if needs_recompute:
        print(f"[INFO] Step 1: Preparing dataset for '{varname}'...")

        ds = xr.open_dataset(infile_path).chunk(CHUNK_DICT_RAW)

        lat_is_coord = 'lat' in ds.coords
        lon_is_coord = 'lon' in ds.coords
        lat_in_vars = 'lat' in ds.data_vars
        lon_in_vars = 'lon' in ds.data_vars

        if lat_is_coord and lon_is_coord:
            print(f"[INFO] lat/lon already set as coordinates. Proceeding.")
        elif lat_in_vars and lon_in_vars:
            print(f"[INFO] Promoting existing lat/lon variables to coordinates...")
            ds = ds.set_coords(['lat', 'lon'])
        else:
            print(f"[INFO] lat/lon not found or invalid. Performing coordinate transformation from E/N...")
            ds.close()
            ds = promote_latlon(infile_path, varname_in_file)

        ds.to_netcdf(step1_path)
        ds.close()
    else:
        print(f"[INFO] Step 1: Skipping (already exists and valid)")

    # 2. HR dataset
    print(f"[INFO] Step 2: Opening high-resolution dataset...")
    highres_ds = xr.open_dataset(step1_path).chunk(CHUNK_DICT_RAW)

    # 3. Coarsen
    step2_path = OUTPUT_DIR / f"{varname}_step2_coarse.nc"
    if not step2_path.exists():
        print(f"[INFO] Step 3: Coarsening dataset for '{varname}'...")
        coarse_ds = conservative_coarsening(highres_ds, varname_in_file, block_size=11)
        coarse_ds.to_netcdf(step2_path)
        coarse_ds.close()
        del coarse_ds
    else:
        print(f"[INFO] Step 3: Skipping (already exists)")

    print(f"[INFO] Step 3: Opening coarsened dataset...")
    coarse_ds = xr.open_dataset(step2_path).chunk(CHUNK_DICT_LATLON)

    # 4. Bicubic
    step3_path = OUTPUT_DIR / f"{varname}_step3_interp.nc"
    if not step3_path.exists():
        print(f"[INFO] Step 4: Interpolating using bicubic remapping...")
        interp_ds = interpolate_bicubic_ds(coarse_ds, highres_ds, varname_in_file).chunk(CHUNK_DICT_LATLON)
        interp_ds.to_netcdf(step3_path)
        interp_ds.close()
        del interp_ds
    else:
        print(f"[INFO] Step 4: Skipping (already exists)")

    print(f"[INFO] Step 4: Opening interpolated dataset...")
    interp_ds = xr.open_dataset(step3_path).chunk(CHUNK_DICT_LATLON)

    # 5. Splitting 80:20 train val
    print(f"[INFO] Step 5: Splitting and scaling dataset...")

    highres = highres_ds[varname_in_file]
    upsampled = interp_ds[varname_in_file].assign_coords(lat=highres_ds['lat'], lon=highres_ds['lon'])
    upsampled['lat'].attrs = highres_ds['lat'].attrs
    upsampled['lon'].attrs = highres_ds['lon'].attrs

    x_train, x_val, y_train, y_val = split(upsampled, highres, TRAIN_RATIO, SEED)

    # 6. Scaling
    print(f"[INFO] Step 6: Computing scaling statistics using CDO...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmpfile:
        x_train.to_netcdf(tmpfile.name)
        stats = get_cdo_stats(tmpfile.name, scale_type)

    x_train_scaled = apply_cdo_scaling(x_train, stats, scale_type)
    x_val_scaled = apply_cdo_scaling(x_val, stats, scale_type)
    y_train_scaled = apply_cdo_scaling(y_train, stats, scale_type)
    y_val_scaled = apply_cdo_scaling(y_val, stats, scale_type)

    # 7. Saving
    print(f"[INFO] Step 7: Saving scaled datasets...")
    x_train_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_input_train_scaled.nc")
    x_val_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_input_val_scaled.nc")
    y_train_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_target_train_scaled.nc")
    y_val_scaled.to_netcdf(OUTPUT_DIR / f"{varname}_target_val_scaled.nc")

    with open(OUTPUT_DIR / f"{varname}_scaling_params.json", "w") as f:
        json.dump(stats, f, indent=2)

    # 8. Cleaning
    print(f"[INFO] Step 8: Cleaning up intermediate files...")
    for step_path in [step1_path, step2_path, step3_path]:
        try:
            os.remove(step_path)
            print(f"[INFO] Deleted {step_path}")
        except FileNotFoundError:
            print(f"[WARN] Could not find {step_path} to delete.")
        except Exception as e:
            print(f"[ERROR] Failed to delete {step_path}: {e}")


    print(f"[INFO] Completed for variable: {varname}")



if __name__ == "__main__":
    print("[INFO] Starting Dask client...")
    client = Client(processes=False)
    ProgressBar().register()

    try:
        main()
    finally:
        client.close()
