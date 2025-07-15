#!/usr/bin/env python3
import os
import subprocess
import config
import xarray as xr

CH_BOX = (5.8, 10.6, 45.7, 47.9)

def process_file(source, target, outname, oldvar, newvar):
    cdo_cmd = [
        "cdo",
        f"sellonlatbox,{CH_BOX[0]},{CH_BOX[1]},{CH_BOX[2]},{CH_BOX[3]}",
        f"-remapbic,{target}",
        source,
        outname
    ]
    print("Running:", " ".join(cdo_cmd))
    subprocess.run(cdo_cmd, check=True)
    print(f"Remapped and cropped: {outname}")

    ds = xr.open_dataset(outname)
    ds = ds.rename({oldvar: newvar})
    ds.to_netcdf(outname, mode="w")
    ds.close()
    print(f"Renamed {oldvar} to {newvar} in {outname}")

pairs = [
    (
        f"{config.MODELS_DIR}/pr_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_PRETRAINING_DIR}/precip_step2_coarse.nc",
        f"{config.MODELS_DIR}/pr_r01_coarse.nc",
        "pr", "precip"
    ),
    (
        f"{config.MODELS_DIR}/tas_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_PRETRAINING_DIR}/temp_step2_coarse.nc",
        f"{config.MODELS_DIR}/tas_r01_coarse.nc",
        "tas", "temp"
    ),
    (
        f"{config.MODELS_DIR}/tasmax_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_PRETRAINING_DIR}/tmax_step2_coarse.nc",
        f"{config.MODELS_DIR}/tmax_r01_coarse.nc",
        "tasmax", "tmax"
    ),
    (
        f"{config.MODELS_DIR}/tasmin_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_PRETRAINING_DIR}/tmin_step2_coarse.nc",
        f"{config.MODELS_DIR}/tmin_r01_coarse.nc",
        "tasmin", "tmin"
    )
]

for src, tgt, out, oldvar, newvar in pairs:
    process_file(src, tgt, out, oldvar, newvar)