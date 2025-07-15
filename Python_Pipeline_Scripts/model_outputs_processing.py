#!/usr/bin/env python3
import xarray as xr
import os
import config  # Contains file paths

def MO_process(source, target, outname, varname):
    if not (os.path.exists(source) and os.path.exists(target)):
        print(f"Missing input: {source} or {target}")
        return

    ds_source = xr.open_dataset(source)
    ds_target = xr.open_dataset(target)

    # Get variable from source and target
    src_var = ds_source[varname]
    tgt_var = list(ds_target.data_vars)[0]  # assumes only one variable in target
    tgt_lat = ds_target['lat']
    tgt_lon = ds_target['lon']

    # Interpolate source to target grid
    src_interp = src_var.interp(
        lat=tgt_lat, lon=tgt_lon, method="cubic"
    )

    # Mask output where target is NaN
    mask = ~xr.ufuncs.isnan(ds_target[tgt_var])
    src_interp = src_interp.where(mask)

    # Save to file
    src_interp.to_netcdf(outname)
    print(f"Saved: {outname}")

file_pairs = [
    (
        f"{config.MODELS_DIR}/pr_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_PRETRAINING_DIR}/precip_step2_coarse.nc",
        f"{config.MODELS_DIR}/pr_r01_cropped.nc",
        "pr"
    ),
    (
        f"{config.MODELS_DIR}/tas_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_PRETRAINING_DIR}/temp_step2_coarse.nc",
        f"{config.MODELS_DIR}/tas_r01_cropped.nc",
        "tas"
    ),
]

for src, tgt, out, var in file_pairs:
    MO_process(src, tgt, out, var)