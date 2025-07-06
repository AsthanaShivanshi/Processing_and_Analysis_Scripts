#!/bin/bash
#SBATCH --job-name=combined_ds_preprocessing
#SBATCH --chdir=/work/FAC/FGSE/IDYST/tbeucler/downscaling
#SBATCH --output=sasthana/Downscaling/Processing_and_Analysis_Scripts/logs/combined_ds_preprocess_%A.log
#SBATCH --error=sasthana/Downscaling/Processing_and_Analysis_Scripts/logs/combined_ds_preprocess_%A.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500G
#SBATCH --time=3-00:00:00

source sasthana/Downscaling/Processing_and_Analysis_Scripts/environment.sh
module load cdo
export PROJ_LIB="$ENVIRONMENT/share/proj"
export HDF5_USE_FILE_LOCKING=FALSE

cd "$BASE_DIR" || { echo "[ERROR] Failed to cd into BASE_DIR"; exit 1; }

VARS=(precip temp tmin tmax)
SCRIPT_PATH="sasthana/Downscaling/Processing_and_Analysis_Scripts/Python_Pipeline_Scripts/processing_combined_ds_1763_2020.py"

export OMP_NUM_THREADS=1

for VAR in "${VARS[@]}"; do
  echo "[INFO] Starting job for variable: $VAR"
  date
  python "$SCRIPT_PATH" --var "$VAR"
  status=$?
  date
  if [ $status -ne 0 ]; then
    echo "[ERROR] Processing failed for variable: $VAR"
    exit 1
  fi
  echo "[INFO] Finished job for variable: $VAR"
done