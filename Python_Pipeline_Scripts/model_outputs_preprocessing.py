import os
import subprocess
import config
import xarray as xr

CH_BOX = (5, 11, 45, 48)

# Masks path
TEMP_MASK_PATH = f"{config.BASE_DIR}/sasthana/Downscaling/Processing_and_Analysis_Scripts/Python_Pipeline_Scripts/temp_mask.nc"
PRECIP_MASK_PATH = f"{config.BASE_DIR}/sasthana/Downscaling/Processing_and_Analysis_Scripts/Python_Pipeline_Scripts/precip_mask.nc"

def process_file(source, target, outname, oldvar, newvar, mask_path):
    outdir = os.path.dirname(outname)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    step1 = outname.replace(".nc", "_step1.nc")
    step2 = outname.replace(".nc", "_step2.nc")

    # remapbic
    cdo_cmd1 = [
        "cdo",
        f"-remapbic,{target}",
        source,
        step1
    ]
    print("Running:", " ".join(cdo_cmd1))
    result1 = subprocess.run(cdo_cmd1, capture_output=True, text=True)
    print(result1.stdout)
    print(result1.stderr)
    if result1.returncode != 0 or not os.path.exists(step1):
        print(f"CDO remapbic failed for {source}")
        return

    # crop
    cdo_cmd2 = [
        "cdo",
        f"sellonlatbox,{CH_BOX[0]},{CH_BOX[1]},{CH_BOX[2]},{CH_BOX[3]}",
        step1,
        step2
    ]
    print("Running:", " ".join(cdo_cmd2))
    result2 = subprocess.run(cdo_cmd2, capture_output=True, text=True)
    print(result2.stdout)
    print(result2.stderr)
    if result2.returncode != 0 or not os.path.exists(step2):
        print(f"CDO sellonlatbox failed for {step1}")
        os.remove(step1)
        return

    # masking and renaming
    ds = xr.open_dataset(step2)
    ds = ds.rename({oldvar: newvar})
    mask_ds = xr.open_dataset(mask_path)
    mask = mask_ds["mask"]
    ds[newvar] = ds[newvar].where(mask)
    ds.to_netcdf(outname, mode="w")
    ds.close()
    print(f"Masked and renamed {oldvar} to {newvar} in {outname}")

    # Intermed files deleted
    os.remove(step1)
    os.remove(step2)
    print(f"Deleted temp files {step1} and {step2}")

pairs = [
    (
        f"{config.MODELS_DIR}/pr_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r2i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_PRETRAINING_DIR}/precip_step2_coarse.nc",
        f"{config.MODELS_DIR}/precip_r02_coarse_masked.nc",
        "pr", "precip", PRECIP_MASK_PATH
    ),
    (
        f"{config.MODELS_DIR}/tas_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r2i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_PRETRAINING_DIR}/temp_step2_coarse.nc",
        f"{config.MODELS_DIR}/temp_r02_coarse_masked.nc",
        "tas", "temp", TEMP_MASK_PATH
    ),
    (
        f"{config.MODELS_DIR}/tasmax_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r2i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_PRETRAINING_DIR}/tmax_step2_coarse.nc",
        f"{config.MODELS_DIR}/tmax_r02_coarse_masked.nc",
        "tasmax", "tmax", TEMP_MASK_PATH
    ),
    (
        f"{config.MODELS_DIR}/tasmin_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r2i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_PRETRAINING_DIR}/tmin_step2_coarse.nc",
        f"{config.MODELS_DIR}/tmin_r02_coarse_masked.nc",
        "tasmin", "tmin", TEMP_MASK_PATH
    )
]

for src, tgt, out, oldvar, newvar, mask_path in pairs:
    process_file(src, tgt, out, oldvar, newvar, mask_path)