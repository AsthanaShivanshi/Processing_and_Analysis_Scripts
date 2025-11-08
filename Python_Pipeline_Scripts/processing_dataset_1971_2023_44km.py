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

np.random.seed(42)


proj_path = os.environ.get("PROJ_LIB") or "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvNew/share/proj"
os.environ["PROJ_LIB"] = proj_path
datadir.set_data_dir(proj_path)

config.GCM_PIPELINE_DIR = Path(config.GCM_PIPELINE_DIR)
config.BASE_DIR = Path(config.BASE_DIR)

CHUNK_DICT_RAW = {"time": 50, "E": 100, "N": 100}
CHUNK_DICT_LATLON = {"time": 50, "lat": 100, "lon": 100}
 
BASE_DIR = Path(os.environ["BASE_DIR"])
INPUT_DIR = BASE_DIR / "sasthana" / "Downscaling"/"Processing_and_Analysis_Scripts" / "data_1971_2023" / "HR_files_full"
OUTPUT_DIR = BASE_DIR / "sasthana" / "Downscaling" / "Downscaling_Models" / "Training_Dataset_50km_SR_1971_2023"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_chunk_dict(ds): #Had to be included due to memory issues. 
    dims = set(ds.dims)
    if {"lat", "lon"}.issubset(dims):
        return CHUNK_DICT_LATLON
    elif {"E", "N"}.issubset(dims):
        return CHUNK_DICT_RAW
    else:
        raise ValueError(f"Dataset has unknown dimensions: {ds.dims}")
    



def promote_latlon(infile, varname):


    with xr.open_dataset(infile).chunk({"time": 50, "N": 100, "E": 100}) as ds:
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
    




def interpolate_bicubic_shell(coarse_ds, target_ds, varname):
    with tempfile.TemporaryDirectory() as tmpdir:
        coarse_file = Path(tmpdir) / "coarse.nc"
        target_file = Path(tmpdir) / "target.nc"
        output_file = Path(tmpdir) / "interp.nc"

        coarse_ds[[varname]].transpose("time", "lat", "lon").to_netcdf(coarse_file)
        target_ds[[varname]].transpose("time", "lat", "lon").to_netcdf(target_file)

        script_path = BASE_DIR / "sasthana" / "Downscaling" / "Processing_and_Analysis_Scripts" / "Python_Pipeline_Scripts" / "bicubic_interpolation.sh"
        try:
            subprocess.run([
                str(script_path), str(coarse_file), str(target_file), str(output_file)
            ], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"interp failed {e}")
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
        # mean and std dev after transform
        epsilon = 1e-3
        stats["epsilon"] = epsilon
        # Use cdo expr to log-transform and then standardise
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



    infile, scale_type, varname_in_file = dataset_map[varname]
    infile_path = INPUT_DIR / infile


    step1_path = OUTPUT_DIR / f"{varname}_step1_latlon.nc"

    if not step1_path.exists() or not {'lat', 'lon'}.issubset(xr.open_dataset(step1_path).coords):
        print(f"[INFO] Step 1: Preparing dataset for '{varname}'")
        with xr.open_dataset(infile_path) as ds:
            ds = ds.chunk(get_chunk_dict(ds))
            if 'lat' in ds.coords and 'lon' in ds.coords:
                pass
            elif 'lat' in ds.data_vars and 'lon' in ds.data_vars:
                ds = ds.set_coords(['lat', 'lon'])
            else:
                ds = promote_latlon(infile_path, varname_in_file)
            # Handling potential negative vals for RhiresD due to dirty data
            if varname == "RhiresD": #cleaning for precip, removing per chance negative values
                ds[varname_in_file] = xr.where(ds[varname_in_file] < 0, 0, ds[varname_in_file])
            ds.to_netcdf(step1_path)


    with xr.open_dataset(step1_path) as highres_ds:
        highres_ds = highres_ds.chunk(get_chunk_dict(highres_ds))

        eurocordex_grid_path = config.GCM_PIPELINE_DIR / "EUROCORDEX_44"/ "tas"/ "tas_day_EUR-44_CLMcom-CCLM4-8-17_MOHC-HadGEM2-ES_r1i1p1_rcp85_1971-2099.nc"
        with xr.open_dataset(eurocordex_grid_path) as eurocordex_grid:
            target_lat = eurocordex_grid['lat']
            target_lon = eurocordex_grid['lon']

        # Interpolate the coarsened data to the target grid
        coarse_interp = highres_ds[varname_in_file].interp(
            lat=target_lat, lon=target_lon, method="linear"
        )

        temp_mask_path = config.BASE_DIR / "sasthana/Downscaling/Processing_and_Analysis_Scripts/Python_Pipeline_Scripts/temp_mask.nc"
        precip_mask_path = config.BASE_DIR / "sasthana/Downscaling/Processing_and_Analysis_Scripts/Python_Pipeline_Scripts/precip_mask.nc"
        mask_file = temp_mask_path if varname in ["TabsD", "TminD", "TmaxD"] else precip_mask_path

        with xr.open_dataset(mask_file) as mask_ds:
            mask = mask_ds['mask'] if 'mask' in mask_ds else list(mask_ds.data_vars.values())[0]
            mask = mask.interp(lat=target_lat, lon=target_lon, method="nearest")

        coarse_interp_masked = coarse_interp.where(mask > 0)

        step2_masked_path = OUTPUT_DIR / f"{varname}_step2_eur44_coarsened.nc"
        coarse_interp_masked.to_netcdf(step2_masked_path)

 
    with xr.open_dataset(step2_masked_path) as coarse_ds:
        coarse_ds = coarse_ds.chunk(get_chunk_dict(coarse_ds))

        step3_path = OUTPUT_DIR / f"{varname}_step3_interp.nc"
        if not step3_path.exists():

            with xr.open_dataset(step1_path) as highres_ds_for_interp:
                highres_ds_for_interp = highres_ds_for_interp.chunk(get_chunk_dict(highres_ds_for_interp))
                interp_ds = interpolate_bicubic_shell(coarse_ds, highres_ds_for_interp, varname_in_file)
                interp_ds = interp_ds.chunk(get_chunk_dict(interp_ds))
                interp_ds.to_netcdf(step3_path)
                interp_ds.close()


    with xr.open_dataset(step3_path) as interp_ds, xr.open_dataset(step1_path) as highres_ds:
        interp_ds = interp_ds.chunk(get_chunk_dict(interp_ds))
        highres_ds = highres_ds.chunk(get_chunk_dict(highres_ds))

        # Chron split: 1971–2010 train, 2011–2020 val, 2021–2023 test
        highres = highres_ds[varname_in_file].sel(time=slice("1971-01-01", "2023-12-31"))
        upsampled = interp_ds[varname_in_file].sel(time=slice("1971-01-01", "2023-12-31"))

        years = np.array(upsampled['time.year'].values)

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
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmpfile:
            y_train.to_netcdf(tmpfile.name)
            stats = get_cdo_stats(tmpfile.name, scale_type, varname_in_file)
        os.remove(tmpfile.name)

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