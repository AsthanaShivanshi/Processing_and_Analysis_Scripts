#!/usr/bin/env python3
import os
import subprocess
import config

def cdo_remapbic(source, target, outname):
    if not (os.path.exists(source) and os.path.exists(target)):
        print(f"Missing input: {source} or {target}")
        return
    cmd = [
        "cdo", f"remapbic,{target}", source, outname
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Saved: {outname}")

def ncrename_var(ncfile, oldvar, newvar):
    cmd = [
        "ncrename", f"-v{oldvar},{newvar}", ncfile
    ]
    print("Renaming variable:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Renamed {oldvar} to {newvar} in {ncfile}")

file_pairs = [
    (
        f"{config.MODELS_DIR}/pr_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_PRETRAINING_DIR}/precip_step2_coarse.nc",
        f"{config.MODELS_DIR}/pr_r01_coarsed_notmasked.nc",
        "pr", "precip"
    ),
    (
        f"{config.MODELS_DIR}/tas_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_PRETRAINING_DIR}/temp_step2_coarse.nc",
        f"{config.MODELS_DIR}/tas_r01_coarsed_notmasked.nc",
        "tas", "temp"
    ),
    (
        f"{config.MODELS_DIR}/tasmax_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_PRETRAINING_DIR}/tmax_step2_coarse.nc",
        f"{config.MODELS_DIR}/tmax_r01_coarsed_notmasked.nc",
        "tasmax", "tmax"
    ),
    (
        f"{config.MODELS_DIR}/tasmin_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_PRETRAINING_DIR}/tmin_step2_coarse.nc",
        f"{config.MODELS_DIR}/tmin_r01_coarsed_notmasked.nc",
        "tasmin", "tmin"
    )
]

for src, tgt, out, oldvar, newvar in file_pairs:
    cdo_remapbic(src, tgt, out)
    ncrename_var(out, oldvar, newvar)