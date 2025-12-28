#!/bin/bash
#SBATCH --job-name=xxkm_preprocessing_obs_data
#SBATCH --chdir=/work/FAC/FGSE/IDYST/tbeucler/downscaling
#SBATCH --output=sasthana/Downscaling/Processing_and_Analysis_Scripts/logs/chron_split_observational_%A_%a.log
#SBATCH --error=sasthana/Downscaling/Processing_and_Analysis_Scripts/logs/chron_split_observational_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4       
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --array=0-3%1          

source sasthana/Downscaling/Processing_and_Analysis_Scripts/environment.sh 

module load cdo

export PROJ_LIB="$ENVIRONMENT/share/proj"
export HDF5_USE_FILE_LOCKING=FALSE 


cd "$BASE_DIR" || { echo "failed to cd into base dir"; exit 1; }

VARS=(RhiresD TabsD TminD TmaxD)

VAR=${VARS[$SLURM_ARRAY_TASK_ID]}

SCRIPT_PATH_1="sasthana/Downscaling/Processing_and_Analysis_Scripts/Python_Pipeline_Scripts/processing_dataset_1971_2023_12km.py"
#SCRIPT_PATH_2="sasthana/Downscaling/Processing_and_Analysis_Scripts/Python_Pipeline_Scripts/processing_dataset_1971_2023_24km.py"
#SCRIPT_PATH_3="sasthana/Downscaling/Processing_and_Analysis_Scripts/Python_Pipeline_Scripts/processing_dataset_1971_2023_36km.py"
#SCRIPT_PATH_4="sasthana/Downscaling/Processing_and_Analysis_Scripts/Python_Pipeline_Scripts/processing_dataset_1971_2023_48km.py"

echo "starting var processing : $VAR"

export OMP_NUM_THREADS=1 

python "$SCRIPT_PATH_1" --var "$VAR"
python "$SCRIPT_PATH_2" --var "$VAR"
python "$SCRIPT_PATH_3" --var "$VAR"
python "$SCRIPT_PATH_4" --var "$VAR"

echo "finished var processing : $VAR"

