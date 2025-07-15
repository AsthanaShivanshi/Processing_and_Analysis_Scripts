#!/usr/bin/env python3
import subprocess
import os
import sys
import config #Contains file paths

def MO_process(source, target, outname):
    interp_file = outname.replace('.nc', '_interp.nc')
    if not (os.path.exists(source) and os.path.exists(target)):
        print(f"Missing input: {source} or {target}")
        return
    cmd_interp = [
        "cdo", f"remapbic,{target}", source, interp_file
    ]
    print("Running:", " ".join(cmd_interp))
    subprocess.run(cmd_interp, check=True)
    if not os.path.exists(interp_file) or os.path.getsize(interp_file) == 0:
        print(f"Remap failed, {interp_file} not created or empty.")
        return
    cmd_mask = [
        "cdo", f"ifthen,{target}", interp_file, outname
    ]
    print("Running:", " ".join(cmd_mask))
    subprocess.run(cmd_mask, check=True)
    os.remove(interp_file)

file_pairs = [
    (
        f"{config.MODELS_DIR}/pr_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc",  #  (pr)
        "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Pretraining_Chronological_Dataset/precip_step2_coarse.nc",  #  (precip)
        f"{config.MODELS_DIR}/pr_r01_cropped.nc"
    ),
    (
        f"{config.MODELS_DIR}/tas_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc",  #  (tas)
        "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Pretraining_Chronological_Dataset/temp_step2_coarse.nc",  #  (temp)
        f"{config.MODELS_DIR}/tas_r01_cropped.nc"
    ),
]

for src, tgt, out in file_pairs:
    MO_process(src, tgt, out)