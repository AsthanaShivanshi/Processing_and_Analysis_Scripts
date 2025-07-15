#!/usr/bin/env python3
import subprocess
import os
import sys
import config #For file paths (HIDDEN)

def MO_process(source, target, outname):
    interp_file = outname.replace('.nc', '_interp.nc')
    cmd_interp = [
        "cdo", f"remapbic,{target}", source, interp_file
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
        f"{config.MODELS_DIR}/source1.nc",
        f"{config.DATASETS_TRAINING_DIR}/RhiresD_step2_coarse.nc",
        f"{config.MODELS_DIR}/source1_matched.nc"
    ),
    (
        f"{config.MODELS_DIR}/source2.nc",
        f"{config.DATASETS_TRAINING_DIR}/TabsD_step2_coarse.nc",
        f"{config.MODELS_DIR}/source2_matched.nc"
    ),
    # Add more (source, target, output) as needed
]

for src, tgt, out in file_pairs:
    MO_process(src, tgt, out)