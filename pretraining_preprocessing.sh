#!/bin/bash
#SBATCH --job-name=pretrain_dask
#SBATCH --chdir=/work/FAC/FGSE/IDYST/tbeucler/downscaling
#SBATCH --output=sasthana/Downscaling/Processing_and_Analysis_Scripts/logs/pretraining_%A_%a.log
#SBATCH --error=sasthana/Downscaling/Processing_and_Analysis_Scripts/logs/pretraining_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4       
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-3%1          

source sasthana/Downscaling/Processing_and_Analysis_Scripts/environment.sh 
export PROJ_LIB="$ENVIRONMENT/share/proj"

cd "$BASE_DIR" || { echo "[ERROR] Failed to cd into BASE_DIR"; exit 1; }

VARS=(precip temp tmin tmax)
VAR=${VARS[$SLURM_ARRAY_TASK_ID]}

SCRIPT_PATH="sasthana/Downscaling/Processing_and_Analysis_Scripts/Python_Pipeline_Scripts/regridding_pretraining_dataset.py"

if [ ! -f "$SCRIPT_PATH" ]; then
  echo "[ERROR] Python script not found: $SCRIPT_PATH"
  exit 1
fi

echo "[INFO] Starting job for variable: $VAR at $(date)"

python "$SCRIPT_PATH" --var "$VAR"

echo "[INFO] Finished job for variable: $VAR at $(date)"
