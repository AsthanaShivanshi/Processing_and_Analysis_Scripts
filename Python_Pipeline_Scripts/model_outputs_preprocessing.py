import os
import subprocess
import config
import xarray as xr

CH_BOX = (5, 11, 45, 48)

def process_file(source, target, outname, oldvar, newvar):
    outdir = os.path.dirname(outname)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    tmpfile = outname.replace(".nc", "_step1.nc")

    cdo_cmd = [
        "cdo",
        f"sellonlatbox,{CH_BOX[0]},{CH_BOX[1]},{CH_BOX[2]},{CH_BOX[3]}",
        f"-remapbic,{target}",
        source,
        tmpfile
    ]
    print("Running:", " ".join(cdo_cmd))
    result = subprocess.run(cdo_cmd, capture_output=True, text=True)
    ds = xr.open_dataset(tmpfile)
    ds = ds.rename({oldvar: newvar})
    ds.to_netcdf(outname, mode="w")
    ds.close()
    print(f"Renamed {oldvar} to {newvar} in {outname}")

    os.remove(tmpfile)
    print(f"Deleted tempfile {tmpfile}")

pairs = [
    (
        f"{config.MODELS_DIR}/pr_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r2i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_PRETRAINING_DIR}/precip_step2_coarse.nc",
        f"{config.MODELS_DIR}/precip_r02_coarse.nc",
        "pr", "precip"
    ),
    (
        f"{config.MODELS_DIR}/tas_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r2i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_PRETRAINING_DIR}/temp_step2_coarse.nc",
        f"{config.MODELS_DIR}/temp_r02_coarse.nc",
        "tas", "temp"
    ),
    (
        f"{config.MODELS_DIR}/tasmax_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r2i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_PRETRAINING_DIR}/tmax_step2_coarse.nc",
        f"{config.MODELS_DIR}/tmax_r02_coarse.nc",
        "tasmax", "tmax"
    ),
    (
        f"{config.MODELS_DIR}/tasmin_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r2i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_PRETRAINING_DIR}/tmin_step2_coarse.nc",
        f"{config.MODELS_DIR}/tmin_r02_coarse.nc",
        "tasmin", "tmin"
    )
]

for src, tgt, out, oldvar, newvar in pairs:
    process_file(src, tgt, out, oldvar, newvar)