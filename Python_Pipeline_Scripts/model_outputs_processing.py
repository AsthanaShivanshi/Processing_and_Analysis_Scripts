#!/usr/bin/env python3
import subprocess
import os
import sys
import config #For file paths (HIDDEN)

def MO_process(source, target, outname):
    interp_file = outname.replace('.nc', '_interp.nc')
    cmd_interp = [
        "cdo", f"remapbil,{target}", source, interp_file
    ]
    print("Running:", " ".join(cmd_interp))
    subprocess.run(cmd_interp, check=True)
#Masking where target is not NaN
    cmd_mask = [
        "cdo", f"ifthen,{target}", interp_file, outname
    ]
    print("Running:", " ".join(cmd_mask))
    subprocess.run(cmd_mask, check=True)

    os.remove(interp_file)

file_pairs = [
    # (source_file, target_file, output_file)
    (
        f"{config.MODELS_DIR}/pr_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_TRAINING_DIR}/RhiresD_step2_coarse.nc",
        f"{config.MODELS_DIR}/pr_r01_cropped.nc"
    ),
    (
        f"{config.MODELS_DIR}/tas_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_TRAINING_DIR}/TabsD_step2_coarse.nc",
        f"{config.MODELS_DIR}/tas_r01_cropped.nc"
    ),
    # Add more (source, target, output) as needed
]

for src, tgt, out in file_pairs:
    MO_process(src, tgt, out)