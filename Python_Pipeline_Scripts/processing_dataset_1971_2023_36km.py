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
import config
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
import joblib

np.random.seed(42)

proj_path = os.environ.get("PROJ_LIB") or "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvNew/share/proj"

os.environ["PROJ_LIB"] = proj_path
datadir.set_data_dir(proj_path)

CHUNK_DICT_RAW = {"time": 50, "E": 100, "N": 100}
CHUNK_DICT_LATLON = {"time": 50, "lat": 100, "lon": 100}

BASE_DIR = Path(os.environ["BASE_DIR"])
INPUT_DIR = BASE_DIR / "sasthana" / "Downscaling"/"Processing_and_Analysis_Scripts" / "data_1971_2023" / "HR_files_full"

OUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Dataset_Setup_I_Chronological_36km"


OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_chunk_dict(ds):
    dims = set(ds.dims)
    if {"lat", "lon"}.issubset(dims):
        return CHUNK_DICT_LATLON
    elif {"E", "N"}.issubset(dims):
        return CHUNK_DICT_RAW
    else:
        raise ValueError(f"Dataset has unknown dimensions: {ds.dims}") #debug

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
    # Area-weighted: ** data by cell Area, sum, and normalised by valid area
    weighted = da.fillna(0) * area
    valid_area = area * da.notnull()
    coarsen_dims = {dim: block_size for dim in ['N', 'E'] if dim in da.dims}
    weighted_sum = weighted.coarsen(**coarsen_dims, boundary='trim').sum(skipna=True)
    area_sum = valid_area.coarsen(**coarsen_dims, boundary='trim').sum(skipna=True)
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
        
        working_dir = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "Python_Pipeline_Scripts"
        
        os.chmod(script_path, 0o755)
        
        subprocess.run([
            str(script_path), str(coarse_file), str(target_file), str(output_file)
        ], check=True, cwd=str(working_dir))
        
        return xr.open_dataset(output_file)[[varname]]


#Global standardisation : temp and precip


def get_stats(da, method):
    arr_flat = da.values.flatten()
    arr_flat = arr_flat[~np.isnan(arr_flat)]
    stats = {}
    if method == "standard":
        stats['mean'] = float(np.mean(arr_flat))
        stats['std'] = float(np.std(arr_flat))

    elif method == "minmax":
        stats['min'] = float(np.min(arr_flat))
        stats['max'] = float(np.max(arr_flat))

    elif method == "log":


        epsilon = float(da.where(da > 0).min().compute().item()) * 0.5
        stats["epsilon"] = epsilon
        arr_flat_log = np.log(arr_flat + epsilon)
        stats['mean'] = float(np.mean(arr_flat_log))
        stats['std'] = float(np.std(arr_flat_log))

    else:
        raise ValueError ("Invalid scaling type")

    return stats



def apply_scaling(da, stats, method):

    
    if method == "standard":
        return (da - stats['mean']) / stats['std']
    elif method == "minmax":
        return (da - stats['min']) / (stats['max'] - stats['min'])
    


    elif method == "log":
        log_da = np.log(da + stats["epsilon"])
        return (log_da - stats['mean']) / stats['std']
    else:
        raise ValueError("Available: z, minmax, log, yeojohnson")
    


def get_stats_sklearn_yeojohnson(da):
    arr_flat = da.values.flatten()
    arr_flat = arr_flat[~np.isnan(arr_flat)]
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    arr_flat_reshaped = arr_flat.reshape(-1, 1)
    pt.fit(arr_flat_reshaped)
    stats = {'lambda': float(pt.lambdas_[0])}
    return stats, pt


def apply_sklearn_yeojohnson(da, pt):
    arr = da.values
    arr_out = np.full_like(arr, np.nan, dtype=np.float64)
    mask = ~np.isnan(arr)
    arr_out[mask] = pt.transform(arr[mask].reshape(-1, 1)).flatten()
    out_da = xr.DataArray(arr_out, dims=da.dims, coords=da.coords, name=da.name)
    return out_da



def get_stats_sklearn_quantile(da):
    arr_flat = da.values.flatten()
    arr_flat = arr_flat[~np.isnan(arr_flat)]
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    arr_flat_reshaped = arr_flat.reshape(-1, 1)
    qt.fit(arr_flat_reshaped)
    return qt

def apply_sklearn_quantile(da, qt):
    arr = da.values
    arr_out = np.full_like(arr, np.nan, dtype=np.float64)
    mask = ~np.isnan(arr)
    arr_out[mask] = qt.transform(arr[mask].reshape(-1, 1)).flatten()
    out_da = xr.DataArray(arr_out, dims=da.dims, coords=da.coords, name=da.name)
    return out_da



def save_split(x_train, y_train, x_val, y_val, x_test, y_test, stats, outdir, varname, pt=None, qt=None): #Signature didnt have QT!!
    if varname == "RhiresD" and pt is not None:


        x_train.to_netcdf(outdir / f"{varname}_input_train_scaled_yeojohnson.nc")
        y_train.to_netcdf(outdir / f"{varname}_target_train_scaled_yeojohnson.nc")
        x_val.to_netcdf(outdir / f"{varname}_input_val_scaled_yeojohnson.nc")
        y_val.to_netcdf(outdir / f"{varname}_target_val_scaled_yeojohnson.nc")
        x_test.to_netcdf(outdir / f"{varname}_input_test_scaled_yeojohnson.nc")
        y_test.to_netcdf(outdir / f"{varname}_target_test_scaled_yeojohnson.nc")
        joblib.dump(pt, outdir / f"{varname}_yeojohnson_transformer.joblib")

        with open(outdir / f"{varname}_scaling_params_yeojohnson.json", "w") as f:
            json.dump(stats, f)


    elif varname == "RhiresD" and qt is not None:
        x_train.to_netcdf(outdir / f"{varname}_input_train_scaled_quantile.nc")
        y_train.to_netcdf(outdir / f"{varname}_target_train_scaled_quantile.nc")
        x_val.to_netcdf(outdir / f"{varname}_input_val_scaled_quantile.nc")
        y_val.to_netcdf(outdir / f"{varname}_target_val_scaled_quantile.nc")
        x_test.to_netcdf(outdir / f"{varname}_input_test_scaled_quantile.nc")
        y_test.to_netcdf(outdir / f"{varname}_target_test_scaled_quantile.nc")
        joblib.dump(qt, outdir / f"{varname}_quantile_transformer.joblib")


    else:
        x_train.to_netcdf(outdir / f"{varname}_input_train_scaled.nc")
        y_train.to_netcdf(outdir / f"{varname}_target_train_scaled.nc")
        x_val.to_netcdf(outdir / f"{varname}_input_val_scaled.nc")
        y_val.to_netcdf(outdir / f"{varname}_target_val_scaled.nc")
        x_test.to_netcdf(outdir / f"{varname}_input_test_scaled.nc")
        y_test.to_netcdf(outdir / f"{varname}_target_test_scaled.nc")



        with open(outdir / f"{varname}_scaling_params.json", "w") as f:
            json.dump(stats, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", type=str, required=True)
    args = parser.parse_args()
    varname = args.var

    dataset_map = {
        "RhiresD": ("RhiresD_1971_2023.nc", "log", "RhiresD"),
        "TabsD":   ("TabsD_1971_2023.nc", "standard", "TabsD"),
        "TminD":   ("TminD_1971_2023.nc", "standard", "TminD"),
        "TmaxD":   ("TmaxD_1971_2023.nc", "standard", "TmaxD"),
    }

    infile, scale_type, varname_in_file = dataset_map[varname]
    infile_path = INPUT_DIR / infile

    step1_path = OUT_DIR / f"{varname}_step1_latlon.nc"

    if not step1_path.exists() or not {'lat', 'lon'}.issubset(xr.open_dataset(step1_path).coords):
        print(f"Step 1: prepping '{varname}'")

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
            ds[varname_in_file] = xr.where(ds[varname_in_file] < 0.1, 0, ds[varname_in_file])
        ds.to_netcdf(step1_path)
        ds.close()

    highres_ds = xr.open_dataset(step1_path).chunk(get_chunk_dict(xr.open_dataset(step1_path)))

    step2_path = OUT_DIR / f"{varname}_step2_coarse.nc"
    if not step2_path.exists():
        coarse_ds = conservative_coarsening(highres_ds, varname_in_file, block_size=36)
        if varname == "RhiresD":
            coarse_ds[varname_in_file] = xr.where(coarse_ds[varname_in_file] < 0.0, 0, coarse_ds[varname_in_file])
        coarse_ds.to_netcdf(step2_path)
        coarse_ds.close()
    coarse_ds = xr.open_dataset(step2_path).chunk(get_chunk_dict(xr.open_dataset(step2_path)))

    step3_path = OUT_DIR / f"{varname}_step3_interp.nc"
    if not step3_path.exists():
        interp_ds = interpolate_bicubic_shell(coarse_ds, highres_ds, varname_in_file)
        interp_ds = interp_ds.chunk(get_chunk_dict(interp_ds))
        if varname == "RhiresD":
            interp_ds[varname_in_file] = xr.where(interp_ds[varname_in_file] < 0, 0, interp_ds[varname_in_file])
        interp_ds.to_netcdf(step3_path)
        interp_ds.close()
    interp_ds = xr.open_dataset(step3_path).chunk(get_chunk_dict(xr.open_dataset(step3_path)))

    highres = highres_ds[varname_in_file].sel(time=slice("1971-01-01", "2023-12-31"))
    upsampled = interp_ds[varname_in_file].sel(time=slice("1971-01-01", "2023-12-31"))
    years = upsampled['time.year'].values

    train_mask = (years >= 1971) & (years <= 2000)
    val_mask   = (years >= 2001) & (years <= 2010)
    test_mask  = (years >= 2011) & (years <= 2023)

    x_train = upsampled.isel(time=train_mask)
    y_train = highres.isel(time=train_mask)
    x_val   = upsampled.isel(time=val_mask)
    y_val   = highres.isel(time=val_mask)
    x_test  = upsampled.isel(time=test_mask)
    y_test  = highres.isel(time=test_mask)


    if varname == "RhiresD" and scale_type == "yeojohnson":
        stats, pt = get_stats_sklearn_yeojohnson(y_train)
        x_train_scaled = apply_sklearn_yeojohnson(x_train, pt)
        x_val_scaled = apply_sklearn_yeojohnson(x_val, pt)
        x_test_scaled = apply_sklearn_yeojohnson(x_test, pt)
        y_train_scaled = apply_sklearn_yeojohnson(y_train, pt)
        y_val_scaled = apply_sklearn_yeojohnson(y_val, pt)
        y_test_scaled = apply_sklearn_yeojohnson(y_test, pt)
        pt_to_save = pt

    elif varname == "RhiresD" and scale_type == "quantile":
        qt = get_stats_sklearn_quantile(y_train)
        x_train_scaled = apply_sklearn_quantile(x_train, qt)
        x_val_scaled = apply_sklearn_quantile(x_val, qt)
        x_test_scaled = apply_sklearn_quantile(x_test, qt)
        y_train_scaled = apply_sklearn_quantile(y_train, qt)
        y_val_scaled = apply_sklearn_quantile(y_val, qt)
        y_test_scaled = apply_sklearn_quantile(y_test, qt)
        pt_to_save = None
        qt_to_save = qt
        stats={}

    else:
        stats = get_stats(y_train, scale_type)
        x_train_scaled = apply_scaling(x_train, stats, scale_type)
        x_val_scaled = apply_scaling(x_val, stats, scale_type)
        x_test_scaled = apply_scaling(x_test, stats, scale_type)
        y_train_scaled = apply_scaling(y_train, stats, scale_type)
        y_val_scaled = apply_scaling(y_val, stats, scale_type)
        y_test_scaled = apply_scaling(y_test, stats, scale_type)
        pt_to_save = None
        qt_to_save = None

    save_split(x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, x_test_scaled, y_test_scaled, stats, OUT_DIR, varname, pt=pt_to_save, qt=qt_to_save)



if __name__ == "__main__":
    client = Client(processes=False)
    try:
        main()
    finally:
        client.close()